import matplotlib,os,pickle
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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
        if best: model.save(path)

    def save(self,model,path,n_per_point=None):
        if not n_per_point:
            model.save(path)
        elif self.current_epoch % n_per_point == 0:
            model.save(path.replace(".h5","_"+str(self.current_epoch)+".h5"))

    def make_history_plot(self,path,n_per_point=1,same_plot=True):
        plt.figure(1)
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

