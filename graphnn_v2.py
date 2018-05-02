import tensorflow as tf
from mlp import Mlp

class GraphNN(object):
	def __init__(
		self,
		var,
		mat,
		msg,
		loop,
		name="GraphNN",
		MLP_weight_initializer = tf.contrib.layers.xavier_initializer,
		MLP_bias_initializer = tf.zeros_initializer,
		Cell_activation = tf.nn.relu,
		Msg_activation = tf.nn.relu,
		Msg_last_activation = None
	):
		"""
		Receives three dictionaries: var, mat and msg.

		○ var is a dictionary from variable names to embedding sizes.
			That is: an entry var["V1"] = 10 means that the variable "V1" will have an embedding size of 10.
		
		○ mat is a dictionary from matrix names to variable pairs.
			That is: an entry mat["M"] = ("V1","V2") means that the matrix "M" can be used to mask messages from "V1" to "V2".
		
		○ msg is a dictionary from function names to variable pairs.
			That is: an entry msg["cast"] = ("V1","V2") means that one can apply "cast" to convert messages from "V1" to "V2".
		
		○ loop is a dictionary from variable names to lists of dictionaries:
			{
				"mat": the matrix name which will be used,
				"transpose?": if true then the matrix M will be transposed,
				"fun": transfer function (python function built using tensorflow operations,
				"msg": message name,
				"var": variable name
			}
			If "mat" is None, it will be the identity matrix,
			If "transpose?" is None, it will default to false,
			if "fun" is None, no function will be applied,
			If "msg" is false, no message conversion function will be applied,
			If "var" is false, then [1] will be supplied as a surrogate.
			
			That is: an entry loop["V2"] = [ {"mat":None,"fun":f,"var":"V2"}, {"mat":"M","transpose?":true,"msg":"cast","var":"V1"} ] enforces the following update rule for every timestep:
				V2 ← tf.append( [ f(V2), Mᵀ × cast(V1) ] )
		"""
		self.var = var
		self.mat = mat
		self.msg = msg
		self.loop = {}
		self.none_ones = {}
		for v, f in loop.items():
			self.loop[v] = []
			for f_dict in f:
				for key in f_dict:
					if key not in [ "mat", "transpose?", "fun", "msg", "var" ]:
						raise Exception( "Loop body definition \"{tuple}\" has fields other than the ones allowed!".format( tuple = f ) ) # TODO correct exception type
					#end if
				#end for
				update_dict = {}
				update_dict["mat"] = f_dict["mat"] if "mat" in f_dict else None
				update_dict["transpose?"] = f_dict["transpose?"] if "transpose?" in f_dict else False
				update_dict["fun"] = f_dict["fun"] if "fun" in f_dict else None
				update_dict["msg"] = f_dict["msg"] if "msg" in f_dict else None
				update_dict["var"] = f_dict["var"] if "var" in f_dict else None
				self.loop[v].append( update_dict )
			#end if
				if update_dict["var"] is None:
					self.none_ones[ self.mat[update_dict["mat"]]["vars"][1] ] = True
				#end if
			#end for
		#end for
		self.name = name
		
		for v in self.var:
			if v not in self.loop:
				Exception( "Variable \"{v}\" not being updated in the loop!".format( v = v ) ) # TODO correct exception type
			#end if
		#end for
		for v in self.loop:
			if v not in self.var:
				raise Exception( "Variable \"{v}\" in the loop has not been declared!".format( v = v ) ) # TODO correct exception type
			#end if
		#end for
		for m, vp in self.mat.items():
			v1, v2 = vp["vars"]
			if v1 not in self.var or v2 not in self.var:
				raise Exception( "Matrix multiplies from an undeclared variable! mat {m} ~ {v1}, {v2}".format( m = m, v1 = v1, v2 = v2) ) # TODO correct exception type
			#end if
			if "compute?" not in vp.keys():
				vp["compute?"] = False
			#end
		#end for
		for m, vp in self.msg.items():
			v1, v2 = vp
			if v1 not in self.var or v2 not in self.var:
				raise Exception( "Message maps from an undeclared variable! msg {m} ~ {v1} -> {v2}".format( m = m, v1 = v1, v2 = v2) ) # TODO correct exception type
			#end if
		#end for
		
		# Hyperparameters
		self.MLP_weight_initializer = MLP_weight_initializer
		self.MLP_bias_initializer = MLP_bias_initializer
		self.Cell_activation = Cell_activation
		self.Msg_activation = Msg_activation
		self.Msg_last_activation = Msg_last_activation
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
		for m, vp in self.mat.items():
			if not vp["compute?"]:
				self.matrix_placeholders[m] = tf.sparse_placeholder( tf.float32, shape = [ None, None ], name = m )
			#end
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
				activations = [ self.Msg_activation for _ in range(2) ] + [ self.Msg_last_activation ],
				name = msg,
				name_internal_layers = True,
				kernel_initializer = self.MLP_weight_initializer(),
				bias_initializer = self.MLP_bias_initializer()
			)
		#end for
		# Init matrix MLPs
		self.mat_MLP = {}
		for M, vs in self.mat.items():
			if vs["compute?"]:
				v1, v2 = vs["vars"]
				self.mat_MLP[M] = Mlp(
				layer_sizes = [ max(self.var[v1],self.var[v2]) for _ in range(2) ] + [ 1 ],
				activations = [ self.Msg_activation for _ in range(2) ] + [ self.Msg_last_activation ],
				name = "mat_MLP_{}".format(M),
				name_internal_layers = True,
				kernel_initializer = self.MLP_weight_initializer(),
				bias_initializer = self.MLP_bias_initializer()
			)
			#end
		#end
		return
	#end _init_parameters
	
	def _init_util_vars(self):
		self.num_vars = {}
		for M, vs in sorted(self.mat.items(), key=lambda x: x[1]['compute?']):
			v1, v2 = vs["vars"]
			if v1 not in self.num_vars:
				if vs["compute?"]:
					self.num_vars[v1] = tf.placeholder( tf.int32, shape = (), name = "{}_n".format(v1) )
				else:
					self.num_vars[v1] = tf.shape( self.matrix_placeholders[M], name = "{}_n".format( v1 ) )[0]
				#end
			#end if
			if v2 not in self.num_vars:
				if vs["compute?"]:
					self.num_vars[v2] = tf.placeholder( tf.int32, shape = (), name = "{}_n".format(v2) )
				else:
					self.num_vars[v2] = tf.shape( self.matrix_placeholders[M], name = "{}_n".format( v2 ) )[1]
				#end
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

		# Compute matrices
		for M, vs in self.mat.items():
			if vs["compute?"]:
				v1, v2 = vs["vars"]
				v1s, v2s = states[v1].h, states[v2].h

				tiled_v1s = tf.reshape(v1s, (tf.shape(v1s)[0],self.var[v1]))
				tiled_v1s = tf.tile(tiled_v1s, (1,tf.shape(v2s)[0]))
				tiled_v1s = tf.reshape(tiled_v1s, (tf.shape(v1s)[0]*tf.shape(v2s)[0],self.var[v1]))

				tiled_v2s = tf.reshape(v2s, (tf.shape(v2s)[0],self.var[v2]))
				tiled_v2s = tf.tile(tiled_v2s, (tf.shape(v1s)[0],1))
				tiled_v2s = tf.reshape(tiled_v2s, (tf.shape(v1s)[0]*tf.shape(v2s)[0],self.var[v2]))

				cells	= tf.concat([tiled_v1s,tiled_v2s], axis=1)
				M_cells = self.mat_MLP[M](cells)

				self.matrix_placeholders[M] = tf.reshape(M_cells, (tf.shape(v1s)[0],tf.shape(v2s)[0]))
			#end
		#end

		new_states = {}
		for v1 in self.var:
			inputs = []
			for D in self.loop[v1]:
				# vs ← V ( or [1] )
				vs = states[D["var"]].h if D["var"] is not None else self.none_ones[ self.mat[ D["mat"] ][1] ]
				# f_vs ← f(V)
				f_vs = D["fun"](vs) if D["fun"] is not None else vs
				# msg_f_vs ← msg(f(V))
				msg_f_vs = self._tf_msgs[D["msg"]]( f_vs ) if D["msg"] is not None else f_vs
				# m_msg_f_vs ← M × msg(f(V))
				if D["mat"] is not None and self.mat[D["mat"]]["compute?"]:
					m_msg_f_vs = tf.matmul( self.matrix_placeholders[ D["mat"] ], msg_f_vs, adjoint_a = D["transpose?"] )
				else:
					m_msg_f_vs = tf.sparse_tensor_dense_matmul( self.matrix_placeholders[ D["mat"] ], msg_f_vs, adjoint_a = D["transpose?"] ) if D["mat"] is not None else msg_f_vs
				#end
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