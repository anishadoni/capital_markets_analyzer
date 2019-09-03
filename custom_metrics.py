class Metrics(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        # self.confusion = []
        # self.precision = []

        # self.recall = []
        # self.f1s = []
        # self.kappa = []
        self.auc = []

    def on_epoch_end(self, epoch, logs={}):
        score = np.asarray(self.model.predict(self.validation_data[0]))
        predict = np.round(np.asarray(
            self.model.predict(self.validation_data[0])))
        targ = self.validation_data[1]

        self.auc.append(sklm.roc_auc_score(targ, score))
        # self.confusion.append(sklm.confusion_matrix(targ, predict))
        # self.precision.append(sklm.precision_score(targ, predict, average=None))
        # self.recall.append(sklm.recall_score(targ, predict))
        # self.f1s.append(sklm.f1_score(targ, predict))
        # self.kappa.append(sklm.cohen_kappa_score(targ, predict))

        return
