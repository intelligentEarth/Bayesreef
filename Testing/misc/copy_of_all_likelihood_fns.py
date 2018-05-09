def probabilisticLikelihood(self, reef, core_data, input_v):
    sim_propn = self.run_Model(reef, input_v)
    sim_propn = sim_propn.T
    intervals = sim_propn.shape[0]
    # # Uncomment if noisy synthetic data is required.
    # self.NoiseToData(intervals,sim_propn)
    log_core = np.log(sim_propn)
    log_core[log_core == -inf] = 0
    z = log_core * core_data
    likelihood = np.sum(z)
    diff = self.diffScore(sim_propn,core_data, intervals)
    # rmse = self.rmse(sim_propn, self.core_data)
    return [likelihood, sim_propn, diff]
           
def noiseToSimLikelihood1000s(self, reef, core_data, input_v):
    # updated likelihood with noise added to simulation
    pred_core = self.run_Model(reef, input_v)
    pred_core = pred_core.T
    pred_core_w_noise = np.zeros((pred_core.shape[0], pred_core.shape[1]))
    intervals = pred_core.shape[0]
    for n in range(intervals):
       pred_core_w_noise[n,:] = np.random.multinomial(1000,pred_core[n],size=1)
    pred_core_w_noise = pred_core_w_noise/1000
    z = np.zeros((intervals,self.communities+1))  
    z = pred_core_w_noise * core_data
    loss = np.log(z)
    loss[loss == -inf] = 0
    loss = np.sum(loss)
    diff = self.diff_score(pred_core_w_noise,core_data, intervals)
    loss = np.exp(loss)
    return [loss, pred_core_w_noise, diff]

def noiseToSimLikelihood1s(self, reef, core_data, input_v)
    intervals = pred_core.shape[0]
    pred_core_w_noise = np.zeros((pred_core.shape[0], pred_core.shape[1])) 
    z = np.zeros((intervals,self.communities+1))    
    for n in range(intervals):
        pred_core_w_noise[n,:] = np.random.multinomial(1,pred_core[n],1)
        idx_data = np.argmax(core_data[n,:])
        idx_model = np.argmax(pred_core_w_noise[n,:])
        if ((pred_core_w_noise[n,self.communities] != 1.) and (idx_data == idx_model)): #where sediment !=1 and max proportions are equal:
            z[n,idx_data] = 1
    diff = self.diff_score(z,intervals)
    # rmse = self.rmse(pred_core, core_data)
    print 'shape pred_core_w_noise', pred_core_w_noise
    print 'pred core', pred_core
    z = z + 0.1
    z = z/(1+(1+self.communities)*0.1)
    loss = np.log(z)
    loss = np.sum(loss)

def deterministicLikelihood(self, reef, core_data, input_v):
    #original, deterministic likelihood comparing binary/discrete data with true data
    pred_core = self.run_Model(reef, input_v)
    pred_core = pred_core.T
    intervals = pred_core.shape[0]
    z = np.zeros((intervals,self.communities+1))    
    for n in range(intervals):
        idx_data = np.argmax(core_data[n,:])
        idx_model = np.argmax(pred_core[n,:])
        if ((pred_core[n,self.communities] != 1.) and (idx_data == idx_model)): #where sediment !=1 and max proportions are equal:
            z[n,idx_data] = 1
    same= np.count_nonzero(z)
    same = float(same)/intervals
    diff = (1-same) *100
    # rmse = self.rmse(pred_core, core_data)        
    z = z + 0.1
    z = z/(1+(1+self.communities)*0.1)
    loss = np.log(z)
    # print 'sum of loss:', np.sum(loss)        
    return [np.sum(loss), pred_core, diff]
