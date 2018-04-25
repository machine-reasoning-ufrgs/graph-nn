import tensorflow as tf
import numpy as np

class Mlp(object):
	def __init__(
		self,
		layer_sizes,
		output_size = None,
		activations = None,
		output_activation = None,
		use_bias = True,
		kernel_initializer = None,
		bias_initializer = tf.zeros_initializer(),
		kernel_regularizer = None,
		bias_regularizer = None,
		activity_regularizer = None,
		kernel_constraint = None,
		bias_constraint = None,
		trainable = True,
		name = None,
		name_internal_layers = True
	):
		"""Stacks len(layer_sizes) dense layers on top of each other, with an additional layer with output_size neurons, if specified."""
		self.layers = []
		internal_name = None
		# If object isn't a list, assume it is a single value that will be repeated for all values
		if not isinstance( activations, list ):
			activations = [ activations for _ in layer_sizes ]
		#end if
		# If there is one specifically for the output, add it to the list of layers to be built
		if output_size is not None:
			layer_sizes = layer_sizes + [output_size]
			activations = activations + [output_activation]
		#end if
		for i, params in enumerate( zip( layer_sizes, activations ) ):
			size, activation = params
			if name_internal_layers:
				internal_name = name + "_MLP_layer_{}".format( i + 1 )
			#end if
			new_layer = tf.layers.Dense(
				size,
				activation = activation,
				use_bias = use_bias,
				kernel_initializer = kernel_initializer,
				bias_initializer = bias_initializer,
				kernel_regularizer = kernel_regularizer,
				bias_regularizer = bias_regularizer,
				activity_regularizer = activity_regularizer,
				kernel_constraint = kernel_constraint,
				bias_constraint = bias_constraint,
				trainable = trainable,
				name = internal_name
			)
			self.layers.append( new_layer )
		#end for
	#end __init__
	
	def __call__( self, inputs, *args, **kwargs ):
		outputs = [ inputs ]
		for layer in self.layers:
			outputs.append( layer( outputs[-1] ) )
		#end for
		return outputs[-1]
	#end __call__
#end Mlp

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
		var is a dictionary from variable names to embedding sizes.
			That is: an entry var["V1"] = 10 means that the variable "V1" will have an embedding size of 10.
		mat is a dictionary from matrix names to variable pairs.
			That is: an entry mat["M"] = ("V1","V2") means that the matrix "M" can be used to mask messages from "V1" to "V2".
		msg is a dictionary from function names to variable pairs.
			That is: an entry msg["cast"] = ("V1","V2") means that one can apply "cast" to convert messages from "V1" to "V2".
		loop is a dictionary from variable names to lists of triples ( matrix name,message name,variable name )
			That is: an entry loop["V2"] = [ (None, None, "V2"), ("M","cast","V1") ] creates this relation for every timestep: V2 <- tf.append( [ I x id(V2) , M x cast(V1) ] )
		"""
		self.var = var
		self.mat = mat
		self.msg = msg
		self.loop = loop
		self.name = name
		
		for v in self.var:
			if v not in loop:
				raise Exception( "Variable \"{v}\" not being updated in the loop!".format( v = v ) ) # TODO correct exception type
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
			for m,f,v2 in self.loop[v1]:
				fv2 = self._tf_msgs[f]( states[v2].h ) if f is not None else states[v2].h
				mfv2 = tf.sparse_tensor_dense_matmul( self.matrix_placeholders[m], fv2 ) if m is not None else fv2
				inputs.append( mfv2 )
			#end for
			if len( inputs ) > 1:
				v_inputs = tf.concat( inputs, axis = 1 )
			else:
				v_inputs = inputs[0]
			#end if
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


def dense_to_sparse( M ):
	n, m = M.shape
	M_i = []
	M_v = []
	M_shape = (n,m)
	for i in range( n ):
		for j in range( m ):
			if M[i,j] > 0.5:
				M_i.append( (i,j ) )
				M_v.append( 1 )
			#end if
		#end for
	#end for
	return (M_i,M_v,M_shape)
#end

if __name__ == "__main__":

	d = 8
	dc = 7
	instance_SAT = tf.placeholder( tf.float32, [ None ], name = "instance_SAT" )
	num_vars_on_instance = tf.placeholder( tf.int32, [ None ], name = "instance_n" )
	p = tf.shape( instance_SAT )[0]
	learning_rate = 2e-5
	parameter_l2norm_scaling = 1e-10
	global_norm_gradient_clipping_ratio = 0.65
	L_vote_MLP = Mlp(
			layer_sizes = [ d for _ in range(2) ],
			activations = [ tf.nn.relu for _ in range(2) ],
			output_size = 1,
			name = "L_vote",
			name_internal_layers = True,
			kernel_initializer = tf.contrib.layers.xavier_initializer(),
			bias_initializer = tf.zeros_initializer()
		)
	
	neurosat = GraphNN(
		{
			"L": d,
			"C": dc
		},
		{
			"M": ("L","C"),
			"Mt": ("C","L"),
			"Inv": ("L","L")
		},
		{
			"Lmsg": ("L","C"),
			"Cmsg": ("C","L")
		},
		{
			"L": [("Inv",None,"L"),("M","Cmsg","C")],
			"C": [("Mt","Lmsg","L")]
		},
		name="NeuroSAT"
	)
	n = tf.floordiv( tf.shape( neurosat.matrix_placeholders["M"] )[0], tf.constant( 2 ) )
	
	# Get the last embeddings
	L_n = neurosat.last_states["L"].h
	L_vote = L_vote_MLP( L_n )
	
	###################
	def _vote_while_body(i, p, n_acc, n, n_var_list, predicted_SAT, L_vote):
		# Helper for the amount of variables in this problem
		i_n = n_var_list[i]
		# Gather the positive and negative literals for that problem
		pos_lits = tf.gather( L_vote, tf.range( n_acc, tf.add( n_acc, i_n ) ) )
		neg_lits = tf.gather( L_vote, tf.range( tf.add( n, n_acc ), tf.add( n, tf.add( n_acc, i_n ) ) ) )
		# Concatenate positive and negative literals and average their vote values
		problem_predicted_SAT = tf.reduce_mean( tf.concat( [pos_lits, neg_lits], axis = 1 ) )
		# Update TensorArray
		predicted_SAT = predicted_SAT.write( i, problem_predicted_SAT )
		return tf.add( i, tf.constant( 1 ) ), p, tf.add( n_acc, i_n ), n, n_var_list, predicted_SAT, L_vote
	#end _message_while_body
	
	def _vote_while_cond(i, p, n_acc, n, n_var_list, predicted_sat, L_vote):
		return tf.less( i, p )
	#end _message_while_cond
	###################
			
	predicted_SAT = tf.TensorArray( size = p, dtype = tf.float32 )
	_, _, _, _, _, predicted_SAT, _ = tf.while_loop(
		_vote_while_cond,
		_vote_while_body,
		[ tf.constant( 0, dtype = tf.int32 ), p, tf.constant( 0, dtype = tf.int32 ), n, num_vars_on_instance, predicted_SAT, L_vote ]
	)
	predicted_SAT = predicted_SAT.stack()
	
	predict_costs = tf.nn.sigmoid_cross_entropy_with_logits( labels = instance_SAT, logits = predicted_SAT )
	predict_cost = tf.reduce_mean( predict_costs )
	vars_cost = tf.zeros([])
	tvars = tf.trainable_variables()
	for var in tvars:
		vars_cost = tf.add( vars_cost, tf.nn.l2_loss( var ) )
	#end for
	vars_cost = tf.Print( vars_cost, [tf.shape( vars_cost )], "Vars ")
	predict_cost = tf.Print( predict_cost, [tf.shape( predict_cost )], "Pred ")
	loss = tf.add( predict_cost, tf.multiply( vars_cost, parameter_l2norm_scaling ) )
	optimizer = tf.train.AdamOptimizer( name = "Adam", learning_rate = learning_rate )
	grads, _ = tf.clip_by_global_norm( tf.gradients( loss, tvars ), global_norm_gradient_clipping_ratio )
	train_step = optimizer.apply_gradients( zip( grads, tvars ) )
	
	accuracy = tf.reduce_mean(
		tf.cast(
			tf.equal(
				tf.cast( instance_SAT, tf.bool ),
				tf.cast( tf.round( tf.nn.sigmoid( predicted_SAT ) ), tf.bool )
			)
			, tf.float32
		)
	)
	with tf.Session() as sess:
		sess.run( tf.global_variables_initializer() )
		l = 10
		m = 20
		M = np.random.rand( l,m )
		Mt = np.transpose( M )
		M = dense_to_sparse( M )
		Mt = dense_to_sparse( Mt )
		Inv =  dense_to_sparse( np.random.rand( 10,10 ) )
		print( 
			sess.run(
				predicted_SAT,
				feed_dict = {
					neurosat.matrix_placeholders[ "M" ]: M,
					neurosat.matrix_placeholders[ "Mt" ]: Mt,
					neurosat.matrix_placeholders[ "Inv" ]: Inv,
					neurosat.time_steps: 5,
					instance_SAT: [1],
					num_vars_on_instance: [l/2]
				}
			)
		)
