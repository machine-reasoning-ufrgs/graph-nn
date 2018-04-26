import tensorflow as tf
import numpy as np
import warnings

from mlp import Mlp

class GraphNN(object):
	def __init__(
		self,
		var,
		mat,
		msg,
		loop,
		name="GraphNN"
	):
		"""
		Receives three dictionaries: var, mat and msg.

		○ var is a dictionary from variable names to embedding sizes.
			That is: an entry var["V1"] = 10 means that the variable "V1" will have an embedding size of 10.
		
		○ mat is a dictionary from matrix names to variable pairs.
			That is: an entry mat["M"] = ("V1","V2") means that the matrix "M" can be used to mask messages from "V1" to "V2".
		
		○ msg is a dictionary from function names to variable pairs.
			That is: an entry msg["cast"] = ("V1","V2") means that one can apply "cast" to convert messages from "V1" to "V2".
		
		○ loop is a dictionary from variable names to lists of quadruples:
			(matrix name, tf function, message name, variable name)
		or quintuples:
			(matrix name, tf function, message name, variable name, transpose? )

			That is: an entry loop["V2"] = [ (None,f,None,"V2") ("M",None,"cast","V1",true) ] enforces the following update rule for every timestep:
				V2 ← tf.append( [ f(V2), Mᵀ × cast(V1) ] )

			Note that if a 5th parameter is not supplied, it is overwritten with 'false' (the default is not to transpose the matrix)
		"""
		self.var = var
		self.mat = mat
		self.msg = msg
		self.loop = {}
		self.none_ones = {}
		for v, f in loop.items():
			self.loop[v] = []
			for vals in f:
				if len( vals ) == 4:
					_mat, _f, _msg, _var = vals
					self.loop[v].append( ( _mat, _f, _msg, _var, False ) )
				elif len( vals ) == 5:
					_mat, _f, _msg, _var, _transpose = vals
					self.loop[v].append( ( _mat, _f, _msg, _var, _transpose ) )
				else:
					raise Exception( "Loop body definition \"{tuple}\" isn't a 4-tuple or 5-tuple!".format( tuple = vals ) ) # TODO correct exception type
				#end if
				if _var is None:
					self.none_ones[ self.mat[_mat][1] ] = True
				#end if
			#end for
		#end for
		self.name = name
		
		for v in self.var:
			if v not in self.loop:
				warnings.warn( "Variable \"{v}\" not being updated in the loop!".format( v = v ) ) # TODO correct exception type
			#end if
		#end for
		for v in self.loop:
			if v not in self.var:
				raise Exception( "Variable \"{v}\" in the loop has not been declared!".format( v = v ) ) # TODO correct exception type
			#end if
		#end for
		for m, vp in self.mat.items():
			v1, v2 = vp
			if v1 not in self.var or v2 not in self.var:
				raise Exception( "Matrix multiplies from an undeclared variable! mat {m} ~ {v1}, {v2}".format( m = m, v1 = v1, v2 = v2) ) # TODO correct exception type
			#end if
		#end for
		for m, vp in self.msg.items():
			v1, v2 = vp
			if v1 not in self.var or v2 not in self.var:
				raise Exception( "Message maps from an undeclared variable! msg {m} ~ {v1} -> {v2}".format( m = m, v1 = v1, v2 = v2) ) # TODO correct exception type
			#end if
		#end for
		
		# Hyperparameters
		self.MLP_weight_initializer = tf.contrib.layers.xavier_initializer
		self.MLP_bias_initializer = tf.zeros_initializer
		self.Cell_activation = tf.nn.relu
		self.Msg_activation = tf.nn.relu
		# Build the network
		with tf.variable_scope( self.name ):
			with tf.variable_scope( "placeholders" ) as scope:
				self._init_placeholders()
			#end placeholder scope
			with tf.variable_scope( "parameters" ) as scope:
				self._init_parameters()
			with tf.variable_scope( "utilities" ) as scope:
				self._init_util_vars()
			with tf.variable_scope( "run" ) as scope:
				self._run()
			#end solve scope
		#end SAT_solver scope
	#end __init__
	
	def _init_placeholders(self):
		self.matrix_placeholders = {}
		for m in self.mat:
			self.matrix_placeholders[m] = tf.sparse_placeholder( tf.float32, shape = [ None, None ], name = m )
		#end for
		self.time_steps = tf.placeholder( tf.int32, shape = (), name = "time_steps" )
		return
	#end _init_placeholders()
	
	def _init_parameters(self):
		# Init embeddings
		self._tf_inits = {}
		for v, d in self.var.items():
			self._tf_inits[v] = tf.get_variable( "{}_init".format( v ), [ 1, d ], dtype = tf.float32 )
		#end for
		# Init LSTM cells
		self._tf_cells = {}
		for v, d in self.var.items():
			self._tf_cells[v] = tf.contrib.rnn.LayerNormBasicLSTMCell(
				d,
				activation = self.Cell_activation
			)
		#end for
		# Init Messages
		self._tf_msgs = {}
		for msg, vs in self.msg.items():
			vin, vout = vs
			self._tf_msgs[msg] = Mlp(
				layer_sizes = [ self.var[vin] for _ in range(2) ] + [ self.var[vout] ],
				activations = [ self.Msg_activation for _ in range(2) ] + [ None ],
				name = msg,
				name_internal_layers = True,
				kernel_initializer = self.MLP_weight_initializer(),
				bias_initializer = self.MLP_bias_initializer()
			)
		#end for
		return
	#end _init_parameters
	
	def _init_util_vars(self):
		self.num_vars = {}
		for M, vs in self.mat.items():
			v1, v2 = vs
			if v1 not in self.num_vars:
				self.num_vars[v1] = tf.shape( self.matrix_placeholders[M], name = "{}_n".format( v1 ) )[0]
			#end if
			if v2 not in self.num_vars:
				self.num_vars[v2] = tf.shape( self.matrix_placeholders[M], name = "{}_n".format( v1 ) )[1]
			#end if
		#end for
		self.pack_vars = {}
		self.pack_indexes = {}
		for i, v in enumerate( self.var ):
			self.pack_indexes[i] = v
			self.pack_vars[v] = i
		#end for
		return
	#end _init_util_vars
	
	def _run(self):
		cell_state = {}
		for v, init in self._tf_inits.items():
			cell_h0 = tf.tile( init, [ self.num_vars[v], 1 ] )
			cell_c0 = tf.zeros_like( cell_h0, dtype = tf.float32 )
			cell_state[v] = tf.contrib.rnn.LSTMStateTuple( h = cell_h0, c = cell_c0 )
			if v in self.none_ones:
				self.none_ones[v] = tf.ones( [ self.num_vars[v], 1 ], dtype = tf.float32, name = "1_{}".format( v ) )
			#end if
		#end for
		
		_, _, cell_state = tf.while_loop(
			self._message_while_cond,
			self._message_while_body,
			[ tf.constant(0), self.time_steps, cell_state ]
		)
		self.last_states = cell_state
		return
	#end _run
	
	def _message_while_body(self, t, t_max, states):
		new_states = {}
		for v1 in self.var:
			inputs = []
			for m,f,msg,v2,transpose in self.loop[v1]:
				# vs 			← V
				# f_vs 			← f(V)
				# msg_f_vs 		← msg(f(V))
				# m_msg_f_vs 	← M × msg(f(V))
				vs 	 		= states[v2].h if v2 is not None else self.none_ones[ self.mat[m][1] ]
				f_vs 		= f(vs) if f is not None else vs
				msg_f_vs 	= self._tf_msgs[msg]( f_vs ) if msg is not None else f_vs
				m_msg_f_vs 	= tf.sparse_tensor_dense_matmul( self.matrix_placeholders[m], msg_f_vs, adjoint_a=transpose ) if m is not None else msg_f_vs
				# Finally, append
				inputs.append( m_msg_f_vs )
			#end for
			v_inputs = tf.concat( inputs, axis = 1 )

			with tf.variable_scope( "{}_cell".format( v1 ) ):
				_, v_state = self._tf_cells[ v1 ]( inputs = v_inputs, state = states[v1] )
				new_states[v1] = v_state
			#end cell variable scope
		#end for
		return tf.add( t, tf.constant( 1 ) ), t_max, new_states
	#end _message_while_body
	
	def _message_while_cond(self, t, t_max, states):
		return tf.less( t, t_max )
	#end _message_while_cond
	
	def __call__( self, *args, **kwargs ):
		return self.last_states
	#end __call__
#end GraphNN
