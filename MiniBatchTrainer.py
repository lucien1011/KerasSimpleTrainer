import matplotlib,os,pickle
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import tensorflow as tf

class MiniBatchTrainer(object):
    def __init__(self,max_latent_count=5):
        self.loss_history_dict = {}
        self.screen_divider = "*"*100
        self.latent_count = 0
        self.max_latent_count = max_latent_count
        self.current_epoch = -1

    def add_loss(self,name,loss):
        if name not in self.loss_history_dict:
            self.loss_history_dict[name] = []
        self.loss_history_dict[name].append(loss)

    def add_epoch(self):
        self.current_epoch += 1

    def print_loss(self,n_per_point=1):
        if self.current_epoch % n_per_point == 0:
            print(self.screen_divider)
            loss_str = " ".join(["["+name+"  %s]"%valueList[self.current_epoch] for name,valueList in self.loss_history_dict.items()])
            print("Epoch %d: " % (self.current_epoch)+loss_str)

    def stop_or_not(self,name,loss):
        if name in self.loss_history_dict:
            return len(self.loss_history_dict[name])-self.loss_history_dict[name].index(min(self.loss_history_dict[name])) > self.max_latent_count
        else:
            return True

    def save_if_best(self,name,loss,model,path):
        best = False
        if name in self.loss_history_dict and self.current_epoch:
            if min(self.loss_history_dict[name]) >= loss:
                best = True
        else:
            best = True
        if best: tf.keras.models.save_model(model,path)

    def save(self,model,path,n_per_point=None):
        filename, file_extension = os.path.splitext(path)
        save_format = file_extension.replace(".","")
        if not n_per_point:
            tf.keras.models.save_model(model,path,save_format=save_format,include_optimizer=True)
        elif self.current_epoch % n_per_point == 0:
            tf.keras.models.save_model(model,path.replace(file_extension,"_"+str(self.current_epoch)+file_extension),save_format=save_format,include_optimizer=True)

    def save_optimiser(self,opt,path,n_per_point=None):
        filename, file_extension = os.path.splitext(path)
        save_format = file_extension.replace(".","")
        #weights = opt.get_config()
        weights = opt.get_weights()
        if not n_per_point:
            pickle.dump(weights,open(path,"wb"))
        elif self.current_epoch % n_per_point == 0:
            pickle.dump(weights,open(path.replace(file_extension,"_"+str(self.current_epoch)+file_extension),"wb"))

    def save_gan(self,gan,gen,disc,path,n_per_point=None):
        filename, file_extension = os.path.splitext(path)
        disc.trainable = False
        self.save(gan,filename+"_gan"+file_extension,n_per_point=n_per_point)
        disc.trainable = True
        self.save(gen,filename+"_gen"+file_extension,n_per_point=n_per_point)
        self.save(disc,filename+"_disc"+file_extension,n_per_point=n_per_point)

    def make_history_plot(self,path,n_per_point=1,same_plot=True,log_scale=False):
        plt.figure(1)
        if log_scale: plt.yscale('log')
        for name,loss_history_list in self.loss_history_dict.items(): 
            n_point = len(loss_history_list)
            plt.plot(range(n_point),loss_history_list,label=name,)
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend(loc='best')
            if not same_plot:
                filename, file_extension = os.path.splitext(path)
                plt.savefig(filename+"_"+name+file_extension)
                plt.clf()
        if same_plot:
            plt.savefig(path)
            plt.clf()

    def save_history(self,path):
        pickle.dump(self.loss_history_dict, open(path,"wb"))

