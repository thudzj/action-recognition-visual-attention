import numpy
import sys
import argparse


from src.actrec_image_recognition import train

def main(job_id, params):
    print 'Anything printed here will end up in the output directory for job #%d' % job_id
    print params

    train(dim_out=params['dim_out'][0],
                                        ctx_dim=params['ctx_dim'][0],
                                        dim=params['dim'][0],
                                        n_actions=params['n_actions'][0],
                                        n_layers_att=params['n_layers_att'][0],
                                        n_layers_out=params['n_layers_out'][0],
                                        n_layers_init=params['n_layers_init'][0],
                                        ctx2out=params['ctx2out'][0],
                                        max_epochs=params['max_epochs'][0],
                                        dispFreq=params['dispFreq'][0],
                                        decay_c=params['decay_c'][0],
                                        alpha_c=params['alpha_c'][0],
                                        temperature_inverse=params['temperature_inverse'][0],
                                        lrate=params['learning_rate'][0],
                                        optimizer=params['optimizer'][0], 
                                        batch_size=params['batch_size'][0],
                                        valid_batch_size=params['valid_batch_size'][0],
                                        saveto=params['model'][0],
                                        validFreq=params['validFreq'][0],
                                        dataset=params['dataset'][0], 
                                        use_dropout=params['use_dropout'][0],
                                        reload_=params['reload'][0],
                                        times=params['times'][0]
                             )

if __name__ == '__main__':
    options = {
        'dim_out': [512],		# hidden layer dim for outputs
        'ctx_dim': [1024],		# context vector dimensionality
        'dim': [512],			# the number of LSTM units
        'n_actions': [102],		# number of digits to predict
        'n_layers_att':[1],
        'n_layers_out': [1],
        'n_layers_init': [1],
        'ctx2out': [False],
        'max_epochs': [100],
        'dispFreq': [20],
        'decay_c': [0.00001], 
        'alpha_c': [0.0], 
        'temperature_inverse': [1],
        'learning_rate': [0.0001],
        'optimizer': ['adam'],
        'batch_size': [128],
        'valid_batch_size': [256],
        'model': ['model_obj101.npz'],
        'validFreq': [200],
        'dataset': ['obj101'],
        'use_dropout': [True],
        'reload': [False],
        'times': [30]
    }

    if len(sys.argv) > 1:
        options.update(eval("{%s}"%sys.argv[1]))

    main(0, options)

