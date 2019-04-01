from Modules.DataProcessing import *
from Modules.Models import *
from Modules.NeuralNetwork import *
from Modules.Utils import *
import time

MODELS = (
    AttributeLengthModel,
    AttributeCharachterDistributionModel,
    ArrtributeStructuralInference,
    AttributePresenceModel,
    )

LOG_DIR = './Logs'

def train(f):
    preds = [] #list of predictions
    Utils.path_exists(LOG_DIR, create_if_not=True)
    dat = DataPreprocessing(f)
    # Iterate over each model
    for M in MODELS:
        # Initialize model
        m = M()
        m.learning(dat)
        m.save_parameters(LOG_DIR+"/"+m.name)
        predictions = m.detection(dat)
        accuracy = Utils.compute_accuracy(dat,predictions)
        preds.append(predictions)
        print "Accuracy of",m.name,":",accuracy*100,"%"
    nn = NeuralNetworkModel(
                            n_inputs=len(MODELS),
                            learning_rate=0.02,
                            batch_size=20000,
                            epochs=100,
                            logdir=LOG_DIR,
                            testing_split=0.3
                            )
    nn.process_input(preds,shuffle=True)
    nn.process_output(dat.outputs(),shuffle=True)
    nn.train()
    print "Neural network model training completed"
    nn.save_parameters(LOG_DIR+"/"+nn.name)
    nn.test()

def detect(f):
    Utils.path_exists(LOG_DIR, create_if_not=True)
    # Dataset object
    dat = DataPreprocessing(f)
    # Load models
    ms = [M() for M in MODELS]
    for m in ms:
        m.load_parameters(LOG_DIR+"/"+m.name)
    # Load NN
    nn = NeuralNetworkModel(
                            n_inputs=len(MODELS),
                            testing_split=1.0,
                            learning_rate=1.0
                            )
    nn.load_parameters(LOG_DIR+"/"+nn.name)
    # Iterate forever
    while(True):
        if len(dat.data)>0:
            print "Processing new data"
            preds = [m.detection(dat) for m in ms] #list of predictions
            nn.process_input(preds,shuffle=False)
            nn_pred = nn.detect()
            dat.save_anomalous_urls(nn_pred,LOG_DIR+'/AnomalousURLS.txt')
        else:
            print "Waiting for file to update...."
        dat.open_file_if_exists(dat.fname)
        dat.read_and_label_data(pos=dat.where,nolabel=True)
        time.sleep(1)


if __name__=="__main__":
    args = Utils.parse_args()
    if args.mode=="Train":
        train(args.data)
    elif args.mode=="Detect":
        print "Detection Model Running....."
        detect(args.data)
    else:
        "Please use mode as Train/Detect"
