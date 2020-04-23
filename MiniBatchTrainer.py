import matplotlib,os
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class Loss(object):
    def __init__(self,losses):
        if type(losses) != list:
            self.losses = [losses]
        else:
            self.losses = losses

    def __repr__(self):
        return ", ".join(map(str,self.losses))

    def sum(self):
        return sum(self.losses)

    def __float__(self):
        return self.sum()

    def __lt__(self,other):
        return self.sum() < other

    def __gt__(self,other):
        return self.sum() > other

    def __len__(self):
        return len(self.losses)

    def __getitem__(self,index):
        if index >= len(self):
            raise IndexError
        else:
            return self.losses[index]

    def get_loss_by_index(self,index):
        return self.losses[index]

class MiniBatchTrainer(object):
    def __init__(self,max_latent_count=5):
        self.test_loss_history = []
        self.train_loss_history = []
        self.screen_divider = "*"*100
        self.latent_count = 0
        self.max_latent_count = max_latent_count
        self.current_epoch = -1

    def add_loss(self,train_loss,test_loss):
        self.train_loss_history.append(Loss(train_loss))
        self.test_loss_history.append(Loss(test_loss))
        self.current_epoch += 1

    def print_loss(self,n_per_point=1):
        if self.current_epoch % n_per_point == 0:
            print(self.screen_divider)
            print("Epoch %d: [Train loss %s] [Test loss %s]" % (self.current_epoch,str(self.train_loss_history[self.current_epoch]),str(self.test_loss_history[self.current_epoch])))

    def stop_or_not(self,loss):
        if self.test_loss_history:
            return len(self.test_loss_history)-self.test_loss_history.index(min(self.test_loss_history)) > self.max_latent_count
        else:
            return True

    def save_if_best(self,losses,model,path):
        best = False
        if type(losses) != list:
            loss_list = [losses]
        else:
            loss_list = losses
        if self.test_loss_history and self.current_epoch:
            if min(self.test_loss_history) >= sum(loss_list):
                best = True
        else:
            best = True
        if best: model.save(path)

    def save(self,model,path,n_per_point=None):
        if not n_per_point:
            model.save(path)
        elif self.current_epoch % n_per_point == 0:
            model.save(path.replace(".h5","_"+str(self.current_epoch)+".h5"))

    def make_history_plot(self,path,n_per_point=1,loss_name_list=[]):
        n_plot = len(self.train_loss_history[0])
        for i_plot in range(n_plot):
            plt.figure(1)
            train_loss_history = [self.train_loss_history[i].get_loss_by_index(i_plot) for i in range(len(self.train_loss_history)) if i % n_per_point == 0]
            test_loss_history = [self.test_loss_history[i].get_loss_by_index(i_plot) for i in range(len(self.test_loss_history)) if i % n_per_point == 0]
            n_point = len(test_loss_history)
            tag = str(i_plot) if not loss_name_list else loss_name_list[i_plot]
            plt.plot(range(n_point),train_loss_history,label="loss "+tag+" train",)
            plt.plot(range(n_point),test_loss_history,label="loss "+tag+" test",linestyle='dashed')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend(loc='best')
            filename, file_extension = os.path.splitext(path)
            plt.savefig(filename+"_"+tag+file_extension)
            plt.clf()
