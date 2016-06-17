# Attention-based action recognition

import theano
import theano.tensor as tensor
theano.config.floatX = 'float32'
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import cPickle as pkl
import numpy
import copy
import os
import time

from scipy import optimize, stats
from collections import OrderedDict

import warnings

from obj101 import obj101

'''
Theano shared variables require GPUs, so to
make this code more portable, these two functions
push and pull variables between a shared
variable dictionary and a regular numpy 
dictionary
'''
# push parameters to Theano shared variables
def zipp(params, tparams):
    for kk, vv in params.iteritems():
        tparams[kk].set_value(vv)

# pull parameters from Theano shared variables
def unzip(zipped):
    new_params = OrderedDict()
    for kk, vv in zipped.iteritems():
        new_params[kk] = vv.get_value()
    return new_params

# get the list of parameters: Note that tparams must be OrderedDict
def itemlist(tparams):
    return [vv for kk, vv in tparams.iteritems()]

# dropout
def dropout_layer(state_before, use_noise, trng):
    proj = tensor.switch(use_noise,
                         state_before *
                         trng.binomial(state_before.shape, p=0.5, n=1, dtype=state_before.dtype),
                         state_before * 0.5)
    return proj

# make prefix-appended name
def _p(pp, name):
    return '%s_%s'%(pp, name)

# all parameters
def init_params(options):
    """
    Initialize all parameters
    """
    params = OrderedDict()
    ctx_dim = options['ctx_dim']

    # init_state, init_cell
    for lidx in xrange(1, options['n_layers_init']):
        params = get_layer('ff')[0](options, params, prefix='ff_init_%d'%lidx, nin=ctx_dim, nout=ctx_dim)
    params = get_layer('ff')[0](options, params, prefix='ff_state', nin=ctx_dim, nout=options['dim'])
    params = get_layer('ff')[0](options, params, prefix='ff_memory', nin=ctx_dim, nout=options['dim'])

    # decoder: LSTM - only 1 layer
    params = get_layer('lstm_cond')[0](options, params, prefix='decoder',
                                       nin=options['ctx_dim'], dim=options['dim'],
                                       dimctx=ctx_dim)

    # Prediction
    params = get_layer('ff')[0](options, params, prefix='ff_logit_lstm', nin=options['dim'], nout=options['dim_out'])
    if options['ctx2out']:
        params = get_layer('ff')[0](options, params, prefix='ff_logit_ctx', nin=ctx_dim, nout=options['dim_out'])
    if options['n_layers_out'] > 1:
        for lidx in xrange(1, options['n_layers_out']):
            params = get_layer('ff')[0](options, params, prefix='ff_logit_h%d'%lidx, nin=options['dim_out'], nout=options['dim_out'])
    params = get_layer('ff')[0](options, params, prefix='ff_logit', nin=options['dim_out'], nout=options['n_actions'])

    return params

# initialize Theano shared variables according to the initial parameters
def init_tparams(params):
    tparams = OrderedDict()
    for kk, pp in params.iteritems():
        tparams[kk] = theano.shared(params[kk], name=kk)
    return tparams

# load parameters
def load_params(path, params):
    pp = numpy.load(path)
    for kk, vv in params.iteritems():
        if kk not in pp:
            raise Warning('%s is not in the archive'%kk)
        params[kk] = pp[kk]
    return params

# layers: 'name': ('parameter initializer', 'feedforward')
layers = {'ff': ('param_init_fflayer', 'fflayer'),
          'lstm_cond': ('param_init_lstm_cond', 'lstm_cond_layer'),
          }

def get_layer(name):
    """
    Part of the reason the init is very slow is because,
    the layer's constructor is called even when it isn't needed
    """
    fns = layers[name]
    return (eval(fns[0]), eval(fns[1]))

# some utilities
def ortho_weight(ndim):
    """
    Random orthogonal weights, we take
    the right matrix in the SVD.

    Remember in SVD, u has the same # rows as W
    and v has the same # of cols as W. So we
    are ensuring that the rows are 
    orthogonal. 
    """
    W = numpy.random.randn(ndim, ndim)
    u, _, _ = numpy.linalg.svd(W)
    return u.astype('float32')

def norm_weight(nin,nout=None, scale=0.01, ortho=True):
    """
    Random weights drawn from a Gaussian
    """
    if nout == None:
        nout = nin
    if nout == nin and ortho:
        W = ortho_weight(nin)
    else:
        W = scale * numpy.random.randn(nin, nout)
    return W.astype('float32')

def tanh(x):
    return tensor.tanh(x)

def rectifier(x):
    return tensor.maximum(0., x)

def linear(x):
    return x

# feedforward layer: affine transformation + point-wise nonlinearity
def param_init_fflayer(options, params, prefix='ff', nin=None, nout=None):
    if nin == None:
        nin = options['dim_proj']
    if nout == None:
        nout = options['dim_proj']
    params[_p(prefix,'W')] = norm_weight(nin, nout, scale=0.01)
    params[_p(prefix,'b')] = numpy.zeros((nout,)).astype('float32')

    return params

def fflayer(tparams, state_below, options, prefix='rconv', activ='lambda x: tensor.tanh(x)', **kwargs):
    return eval(activ)(tensor.dot(state_below, tparams[_p(prefix,'W')])+tparams[_p(prefix,'b')])

# Conditional LSTM layer with Attention
def param_init_lstm_cond(options, params, prefix='lstm_cond', nin=None, dim=None, dimctx=None):
    if nin == None:
        nin = options['dim']
    if dim == None:
        dim = options['dim']
    if dimctx == None:
        dimctx = options['dim']

    # input to LSTM
    W = numpy.concatenate([norm_weight(nin,dim),
                           norm_weight(nin,dim),
                           norm_weight(nin,dim),
                           norm_weight(nin,dim)], axis=1)
    params[_p(prefix,'W')] = W

    # LSTM to LSTM
    U = numpy.concatenate([ortho_weight(dim),
                           ortho_weight(dim),
                           ortho_weight(dim),
                           ortho_weight(dim)], axis=1)
    params[_p(prefix,'U')] = U

    # bias to LSTM
    params[_p(prefix,'b')] = numpy.zeros((4 * dim,)).astype('float32')

    # attention: context -> hidden
    Wc_att = norm_weight(dimctx, ortho=False)
    params[_p(prefix,'Wc_att')] = Wc_att

    # attention: LSTM -> hidden
    Wd_att = norm_weight(dim,dimctx)
    params[_p(prefix,'Wd_att')] = Wd_att

    # attention: hidden bias
    b_att = numpy.zeros((dimctx,)).astype('float32')
    params[_p(prefix,'b_att')] = b_att

    # deeper attention
    if options['n_layers_att'] > 1:
        for lidx in xrange(1, options['n_layers_att']):
            params[_p(prefix,'W_att_%d'%lidx)] = ortho_weight(dimctx)
            params[_p(prefix,'b_att_%d'%lidx)] = numpy.zeros((dimctx,)).astype('float32')

    # attention: 
    U_att = norm_weight(dimctx,1)
    params[_p(prefix,'U_att')] = U_att
    c_att = numpy.zeros((1,)).astype('float32')
    params[_p(prefix, 'c_tt')] = c_att

    return params

def lstm_cond_layer(tparams, state_below, options, prefix='lstm',
                    init_memory=None, init_state=None,
                    trng=None, use_noise=None,
                    **kwargs):
    """
    Computation graph for the LSTM.
    Note that we removed 'context' and put this into 'state_below'
    Video frames need to be part of scan, since it changes each step
    """
    nsteps = options["times"]
    n_samples = state_below.shape[0]
    n_annotations = state_below.shape[1]


    dim = tparams[_p(prefix, 'U')].shape[0]

    # initial/previous state
    if init_state == None:
        init_state = tensor.alloc(0., n_samples, dim)
    # initial/previous memory 
    if init_memory == None:
        init_memory = tensor.alloc(0., n_samples, dim)

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n*dim:(n+1)*dim]
        return _x[:, n*dim:(n+1)*dim]

    def _step(h_, c_, a_, ct_, _x):
        # xt, ht-1, ct-1, alpha, ctx
        # attention
        # print '\n\ncheck\n\n'
        pstate_ = tensor.dot(h_, tparams[_p(prefix,'Wd_att')]) # pstate_
        pctx_ = tensor.dot(x_, tparams[_p(prefix,'Wc_att')]) + tparams[_p(prefix, 'b_att')]
        if options['n_layers_att'] > 1:
            for lidx in xrange(1, options['n_layers_att']):
                pctx_ = tensor.dot(pctx_, tparams[_p(prefix,'W_att_%d'%lidx)])+tparams[_p(prefix, 'b_att_%d'%lidx)]
                if lidx < options['n_layers_att'] - 1:
                    pctx_ = tanh(pctx_)
        pctx_ = pctx_ + pstate_[:,None,:]
        #pctx_list = []
        #pctx_list.append(pctx_)
        pctx_ = tanh(pctx_)
        alpha = tensor.dot(pctx_, tparams[_p(prefix,'U_att')])+tparams[_p(prefix, 'c_tt')]
        #alpha_pre = alpha
        alpha_shp = alpha.shape
        alpha = tensor.nnet.softmax(options['temperature_inverse']*alpha.reshape([alpha_shp[0],alpha_shp[1]])) # softmax
        ctx_ = (x_ * alpha[:,:,None]).sum(1) # current context
        # print '\n\ncheck\n\n'


        preact = tensor.dot(h_, tparams[_p(prefix, 'U')])
        preact += tensor.dot(ctx_, tparams[_p(prefix, 'W')]) + tparams[_p(prefix, 'b')]

        i = _slice(preact, 0, dim)              # z_it
        f = _slice(preact, 1, dim)              # z_ft
        o = _slice(preact, 2, dim)              # z_ot
        i = tensor.nnet.sigmoid(i)              # it = sigmoid(z_it)
        f = tensor.nnet.sigmoid(f)              # ft = sigmoid(z_ft)
        o = tensor.nnet.sigmoid(o)              # ot = sigmoid(z_ot)
        c = tensor.tanh(_slice(preact, 3, dim)) # at = tanh(z_at)

        c = f * c_ + i * c                      # ct = ft * ct-1 + it * at

        h = o * tensor.tanh(c)                  # ht = ot * thanh(ct)

        rval = [h, c, alpha, ctx_]
        # rval += [pstate_, pctx_, i, f, o, preact, alpha_pre]+pctx_list
        # print '\n\ncheck\n\n'
        return rval


    #_step0 = lambda h_, c_, a_, ct_, x_: _step(h_, c_, a_, ct_, x_)

    noseqs = [state_below]
    outputs_info = [init_state,
                    init_memory,
                    tensor.alloc(0., n_samples, n_annotations),
                    tensor.alloc(0., n_samples, options['ctx_dim'])]
    rval, updates = theano.scan(_step,
                                outputs_info=outputs_info,
																non_sequences = noseqs,
                                name=_p(prefix, '_layers'),
                                n_steps=nsteps, profile=False)
    return rval

# build a training model
def build_model(tparams, options):
    """
    Build up the whole computation graph
    """
    trng = RandomStreams(1234)
    use_noise = theano.shared(numpy.float32(0.))
    # video blocks. (n_timesteps, n_samples, n_annotations, ctxdim)
    x = tensor.tensor3('x', dtype='float32')
    n_timesteps = options["times"]
    n_samples = x.shape[0]
    n_annotations = x.shape[1]
    ctxdim = x.shape[2]
  
    # action labels
    y = tensor.matrix('y', dtype='int64')
    ctx = x

    # initial state/cell
    ctx_mean = ctx.mean(1) ### you want ctx_mean to be n_samples x ctxdim

    for lidx in xrange(1, options['n_layers_init']):
        ctx_mean = get_layer('ff')[1](tparams, ctx_mean, options,
                                      prefix='ff_init_%d'%lidx, activ='rectifier')
        if options['use_dropout']:
            ctx_mean = dropout_layer(ctx_mean, use_noise, trng)

    init_state = get_layer('ff')[1](tparams, ctx_mean, options, prefix='ff_state', activ='tanh')
    init_memory = get_layer('ff')[1](tparams, ctx_mean, options, prefix='ff_memory', activ='tanh')

    # decoder
    proj = get_layer('lstm_cond')[1](tparams, ctx, options,
                                     prefix='decoder',
                                     init_state=init_state,
                                     init_memory=init_memory,
                                     trng=trng,
                                     use_noise=use_noise)
    # collection
    proj_h = proj[0][-1]
    alphas = proj[2]
    ctxs = proj[3]
    if options['use_dropout']:
        proj_h = dropout_layer(proj_h, use_noise, trng)

    # outputs
    logit = get_layer('ff')[1](tparams, proj_h, options, prefix='ff_logit_lstm', activ='linear')
    if options['ctx2out']:
        logit += get_layer('ff')[1](tparams, ctxs, options, prefix='ff_logit_ctx', activ='linear')
    logit = tanh(logit)
    if options['use_dropout']:
        logit = dropout_layer(logit, use_noise, trng)
    if options['n_layers_out'] > 1:
        for lidx in xrange(1, options['n_layers_out']):
            logit = get_layer('ff')[1](tparams, logit, options, prefix='ff_logit_h%d'%lidx, activ='rectifier')
            if options['use_dropout']:
                logit = dropout_layer(logit, use_noise, trng)

    logit = get_layer('ff')[1](tparams, logit, options, prefix='ff_logit', activ='linear')
    #logit_shp = logit.shape #(TS, BS, #actions)

    #logit = logit.reshape([logit_shp[0]*logit_shp[1], logit_shp[2]])  #(TSxBS,#actions)
    probs = tensor.nnet.softmax(logit)                                #(TSxBS,#actions)
    #probs = probs.reshape([logit_shp[0], logit_shp[1], logit_shp[2]]) #(TS,BS,#actions)

    # Predictions
    #preds = probs #(BS,#actions)
    preds = tensor.argmax(probs,axis=1) # computed y; true y is in 'y' #(BS,1)

    # Cost function
    #probs = probs.reshape([logit_shp[0]*logit_shp[1], logit_shp[2]])
    #tmp = tensor.reshape(y, [x.shape[0]*x.shape[1],])
    cost = -tensor.log(probs[tensor.arange(n_samples), y] + 1e-8)
    cost = cost.reshape([n_samples,])
    cost = cost.sum(0)

    opt_outs = dict()

    return trng, use_noise, [x, y], alphas, cost, opt_outs, preds

def pred_acc(modelname, batch_size, f_preds, data_test_pb, dh_test, test_dataset_size, num_test_batches, test=True, verbose=False):
    """
    Make predictions for new data
    """
    dh_test.Reset()
    n_examples = test_dataset_size
    preds = numpy.zeros((n_examples,)).astype('float32')
    n_done = 0

    for tbidx in xrange(num_test_batches):
        n_done += batch_size
        x, y, n_ex = dh_test.GetBatch(data_test_pb)
        pred_ = f_preds(x,y)
        if n_ex == batch_size:
            preds[tbidx*batch_size:tbidx*batch_size+batch_size] = pred_[:]
        else:
            preds[tbidx*batch_size:tbidx*batch_size+n_ex] = pred_[0:n_ex]

        if verbose:
            print '%d/%d examples computed'%(n_done,n_examples)

    if test==True:
        fileprefix = 'test_results_'
    else:
        fileprefix = 'train_results_'
    tempfilename = fileprefix + modelname.split('/')[-1].split('.')[0] + '.txt'
    f = open(tempfilename, 'w')
    vid_idx = 0
    resultstr='{} '.format(vid_idx)
    for i in xrange(n_examples):
        if dh_test.video_ind_[dh_test.frame_indices_[i]] == vid_idx:
            resultstr=resultstr+'{},'.format(int(preds[i]))
        else:
            vid_idx = vid_idx+1
            resultstr=resultstr[:-1]+'\n'
            f.write(resultstr)
            resultstr='{} '.format(vid_idx)
            resultstr=resultstr+'{},'.format(int(preds[i]))
    resultstr=resultstr[:-1]+'\n'
    f.write(resultstr)
    f.close()

    f = open(tempfilename,'r')
    lines = f.readlines()
    f.close()

    pred  = numpy.zeros(len(lines)).astype('int64')
    for i in xrange(len(lines)):
        try:
            s=lines[i].split(' ')[1]
            s=s[0:-1]
            s=s.split(',')
            s = [int(x) for x in s]
            s = numpy.array(s)
            s = stats.mode(s)[0][0]
            pred[i] = int(s)
        except IndexError:
            print 'One blank index skipped'
            pred[i] = -1

    f = open(data_test_pb.labels_file,'r')
    lines = f.readlines()
    f.close()
    f = open(data_test_pb.num_frames_file,'r')
    framenum = f.readlines()
    f.close()
    truth  = numpy.zeros(len(lines)).astype('int64')
    framel = numpy.zeros(len(lines)).astype('int64')
    for i in xrange(len(lines)):
        s=lines[i][0:-1]
        truth[i] = int(s)
        framel[i]= int(framenum[i][0:-1])
    return (truth==pred).mean()

# optimizers
# name(hyperp, tparams, grads, inputs (list), cost) = f_grad_shared, f_update
def adam(lr, tparams, grads, inp, cost):
    """
    Adam: A Method for Stochastic Optimization (Diederik Kingma, Jimmy Ba)
    """
    gshared = [theano.shared(p.get_value() * numpy.float32(0.), name='%s_grad'%k) for k, p in tparams.iteritems()]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]
    #print '\n\ncheck\n\n'
    f_grad_shared = theano.function(inp, cost, updates=gsup, allow_input_downcast=True)

    # Magic numbers
    lr0 = 0.0002
    b1 = 0.1
    b2 = 0.001
    e = 1e-8

    updates = []

    i = theano.shared(numpy.float32(0.))
    i_t = i + 1.
    fix1 = 1. - b1**(i_t)
    fix2 = 1. - b2**(i_t)
    lr_t = lr0 * (tensor.sqrt(fix2) / fix1)

    for p, g in zip(tparams.values(), gshared):
        m = theano.shared(p.get_value() * numpy.float32(0.))
        v = theano.shared(p.get_value() * numpy.float32(0.))
        m_t = (b1 * g) + ((1. - b1) * m)
        v_t = (b2 * tensor.sqr(g)) + ((1. - b2) * v)
        g_t = m_t / (tensor.sqrt(v_t) + e)
        p_t = p - (lr_t * g_t)
        updates.append((m, m_t))
        updates.append((v, v_t))
        updates.append((p, p_t))
    updates.append((i, i_t))

    f_update = theano.function([lr], [], updates=updates, on_unused_input='ignore')

    return f_grad_shared, f_update

def adadelta(lr, tparams, grads, inp, cost):
    zipped_grads = [theano.shared(p.get_value() * numpy.float32(0.), name='%s_grad'%k) for k, p in tparams.iteritems()]
    running_up2 = [theano.shared(p.get_value() * numpy.float32(0.), name='%s_rup2'%k) for k, p in tparams.iteritems()]
    running_grads2 = [theano.shared(p.get_value() * numpy.float32(0.), name='%s_rgrad2'%k) for k, p in tparams.iteritems()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2)) for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function(inp, cost, updates=zgup+rg2up, profile=False)

    updir = [-tensor.sqrt(ru2 + 1e-6) / tensor.sqrt(rg2 + 1e-6) * zg for zg, ru2, rg2 in zip(zipped_grads, running_up2, running_grads2)]
    ru2up = [(ru2, 0.95 * ru2 + 0.05 * (ud ** 2)) for ru2, ud in zip(running_up2, updir)]
    param_up = [(p, p + ud) for p, ud in zip(itemlist(tparams), updir)]

    f_update = theano.function([lr], [], updates=ru2up+param_up, on_unused_input='ignore', profile=False)

    return f_grad_shared, f_update

def rmsprop(lr, tparams, grads, inp, cost):
    zipped_grads = [theano.shared(p.get_value() * numpy.float32(0.), name='%s_grad'%k) for k, p in tparams.iteritems()]
    running_grads = [theano.shared(p.get_value() * numpy.float32(0.), name='%s_rgrad'%k) for k, p in tparams.iteritems()]
    running_grads2 = [theano.shared(p.get_value() * numpy.float32(0.), name='%s_rgrad2'%k) for k, p in tparams.iteritems()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rgup = [(rg, 0.95 * rg + 0.05 * g) for rg, g in zip(running_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2)) for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function(inp, cost, updates=zgup+rgup+rg2up, profile=False)

    updir = [theano.shared(p.get_value() * numpy.float32(0.), name='%s_updir'%k) for k, p in tparams.iteritems()]
    updir_new = [(ud, 0.9 * ud - 1e-4 * zg / tensor.sqrt(rg2 - rg ** 2 + 1e-4)) for ud, zg, rg, rg2 in zip(updir, zipped_grads, running_grads, running_grads2)]
    param_up = [(p, p + udn[1]) for p, udn in zip(itemlist(tparams), updir_new)]
    f_update = theano.function([lr], [], updates=updir_new+param_up, on_unused_input='ignore', profile=False)

    return f_grad_shared, f_update

def sgd(lr, tparams, grads, inp, cost):
    gshared = [theano.shared(p.get_value() * numpy.float32(0.), name='%s_grad'%k) for k, p in tparams.iteritems()]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]

    f_grad_shared = theano.function(inp, cost, updates=gsup, profile=False)

    pup = [(p, p - lr * g) for p, g in zip(itemlist(tparams), gshared)]
    f_update = theano.function([lr], [], updates=pup, profile=False)

    return f_grad_shared, f_update

# validate options
def validate_options(options):
    """
    Return warning messages for hyperparams
    """

def train(dim_out=100, # hidden layer dim for outputs
          ctx_dim=512, # context vector dimensionality
          dim=1000, # the number of LSTM units
          n_actions=3, # number of actions to predict
          n_layers_att=1,
          n_layers_out=1,
          n_layers_init=1,
          ctx2out=False,
          max_epochs=5000,
          dispFreq=100,
          decay_c=0.,
          alpha_c=0.,
          temperature_inverse=1.0,
          lrate=0.01,
          optimizer='adam',
          batch_size = 16,
          valid_batch_size = 16,
          saveto='model.npz',
          validFreq=1000,
          dataset='flickr8k', # dummy dataset, replace with video ones
          use_dropout=False,
          reload_=False,
          times=30):

    # Model options
    model_options = locals().copy()
    #model_options = validate_options(model_options)

    # reload options
    if reload_ and os.path.exists(saveto):
        print "Reloading options"
        with open('%s.pkl'%saveto, 'rb') as f:
            model_options = pkl.load(f)

    print 'Loading data start'
    if dataset == "obj101":
			train = obj101("googlenet_train_features_tmp.hkl", "googlenet_train_labels.hkl")
			test = obj101("googlenet_test_features_tmp.hkl", "googlenet_test_labels.hkl")
    print 'Loading data finish'

    print 'Building model'
    params = init_params(model_options)
    # reload parameters
    if reload_ and os.path.exists(saveto):
        print "Reloading model"
        params = load_params(saveto, params)

    tparams = init_tparams(params)

    trng, use_noise, \
          inps, alphas, \
          cost, \
          opts_out, preds = \
          build_model(tparams, model_options)

    # before any regularizer
    f_log_probs = theano.function(inps, -cost, profile=False)
    f_preds = theano.function(inps, [preds, -cost], profile=False, on_unused_input='ignore')

    if decay_c > 0.:
        decay_c = theano.shared(numpy.float32(decay_c), name='decay_c')
        weight_decay = 0.
        for kk, vv in tparams.iteritems():
            weight_decay += (vv ** 2).sum()
        weight_decay *= decay_c
        cost += weight_decay

    if alpha_c > 0.:
        alpha_c = theano.shared(numpy.float32(alpha_c), name='alpha_c')
        alpha_reg = alpha_c * ((1.-alphas.sum(0))**2).sum(1).mean()
        cost += alpha_reg

    # gradient computation
    grads = tensor.grad(cost, wrt=itemlist(tparams))
    lr = tensor.scalar(name='lr')
    f_grad_shared, f_update = eval(optimizer)(lr, tparams, grads, inps, cost)

    print 'Optimization'

    best_p = None
    acc_max = 0

    uidx = 0
    num_train_batches = train.num_examples / batch_size
    num_test_batches = test.num_examples / valid_batch_size

    for epochidx in xrange(max_epochs):
        print 'Epoch ', epochidx
        for tbidx in xrange(num_train_batches):
            uidx += 1
            use_noise.set_value(1.)

            pd_start = time.time()
            x, y = train.next_batch(batch_size)
            pd_duration = time.time() - pd_start

            if x == None:
                print 'Minibatch with zero sample'
                continue
            ud_start = time.time()

            cost = f_grad_shared(x, y)
            f_update(lrate)
            ud_duration = time.time() - ud_start

            if numpy.isnan(cost):
                print 'NaN detected in cost'
                return 1., 1.
            if numpy.isinf(cost):
                print 'INF detected in cost'
                return 1., 1.

            if numpy.mod(uidx, dispFreq) == 0:
                print 'Epoch ', epochidx, 'Update ', uidx, 'Cost ', cost, 'PD ', pd_duration, 'UD ', ud_duration

            if numpy.mod(uidx, validFreq) == 0:
								use_noise.set_value(0.)
								test_err = 0
								print 'Testing'
								cost_list = []
								acc_list = []
								for test_index in xrange(num_test_batches):
									x, y = test.next_batch(valid_batch_size)
									pred_, cost = f_preds(x, y)
									acc = float(sum(pred_ == y.reshape(pred_.shape))) / pred_.shape[0]
									cost_list.append(cost)
									acc_list.append(acc)
								cost_mean = numpy.mean(cost_list)
								acc_mean = numpy.mean(acc_list)
                if uidx == 0 or acc_mean > acc_max:
										best_p = unzip(tparams) # p for min valid err / max valid acc
										acc_max = acc_mean

                print 'Cost ', cost_mean, 'acc', acc_mean

    if best_p is not None:
        zipp(best_p, tparams)
    params = copy.copy(best_p)
    numpy.savez(saveto, zipped_params=best_p, **params)
    pkl.dump(model_options, open('%s.pkl'%saveto, 'wb'))

    print model_options

    return 1, 1

if __name__ == '__main__':
    pass

