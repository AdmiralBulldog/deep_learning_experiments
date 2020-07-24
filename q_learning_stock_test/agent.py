from q_utils import makeNN
import q_utils
import random
import numpy as np
import torch
import torch.nn as nn

class Agent:
    def __init__(self,
                gamma=0.9,
                lambd=0.0001,
                epsilonB=1.0,
                epsilonS=1.0,
                epsilon_min=1.0,#0.01
                epsilon_decay=1.0/10000,
                tgt_net_update_freqB=10000,
                tgt_net_update_freqS=10000,
                input_size=100,
                replay_capacity=1000000,
                batch_size=32,
                device=torch.device('cpu')):
                
        self.gamma = gamma
        self.lambd = lambd
        self.epsilonB = epsilonB
        self.epsilonS = epsilonS
        self.epsilon_min = epsilon_min#0.01
        self.epsilon_decay = epsilon_decay
        self.parameter_updatesB = 0
        self.tgt_net_update_freqB = tgt_net_update_freqB
        self.parameter_updatesS = 0
        self.tgt_net_update_freqS = tgt_net_update_freqS
        self.input_size = input_size
        self.replay_capacity = replay_capacity
        self.batch_size = batch_size
        self.device = device
        self.replayS = q_utils.ExperienceReplay([self.input_size], self.replay_capacity)
        self.replayB = q_utils.ExperienceReplay([self.input_size], self.replay_capacity)
        self.buyNN = q_utils.makeNN(self.input_size,2)
        self.sellNN = q_utils.makeNN(self.input_size,1)
        self.buyNN_tgt = q_utils.makeNN(self.input_size,2)
        self.sellNN_tgt = q_utils.makeNN(self.input_size,1)
        

    def actBuy(self, states):
        if np.random.rand() <= self.epsilonB:
            return random.randrange(2) #[random.randrange(2) for _ in range(batch_size)]
        #print("HERE")
        self.buyNN.eval()
        qs = None
        with torch.no_grad():
            qs = self.buyNN(states.unsqueeze(0))[0]
        self.buyNN.train()    
        return torch.argmax(qs, dim=0)

    def actSell(self, states):
        #batch_size = len(states)
        if np.random.rand() <= self.epsilonS:
            return random.randrange(2) #[random.randrange(2) for _ in range(batch_size)]
        self.sellNN.eval()
        qs = None
        with torch.no_grad():
            qs = self.sellNN(states.unsqueeze(0))[0]
        self.sellNN.train()    
        if qs > 0:
            return torch.tensor(0.0)
        else:
            return torch.tensor(1.0)    

    def learnB(self, optimB):
        optimB.zero_grad()
        states, actions, reward, nextStates = q_utils.sample_memory(self.replayB, batch_size=self.batch_size)
        
        with torch.no_grad():
            target = reward + self.gamma * torch.max(self.buyNN_tgt(nextStates), dim=1)[0]
            #target = reward[0] + gamma * torch.max(buyNN(nextStates[0]), dim=0)[0]
            
        q = self.buyNN(states)[np.arange(self.batch_size), actions]
        loss = torch.sum((target - q)**2)
        
        loss.backward()
        optimB.step()

        return loss
    

    def learnS(self, optimS):
        optimS.zero_grad()
        states, _, reward, nextStates = q_utils.sample_memory(self.replayS, batch_size=self.batch_size)
        with torch.no_grad():
            target = reward + self.gamma * torch.max(self.sellNN_tgt(nextStates).squeeze(), torch.zeros(self.batch_size, device=self.device))
        q = self.sellNN(states).squeeze()

        loss = torch.sum((target - q)**2)
        
        loss.backward()
        optimS.step()

        return loss        

    def learn(self, data, epochs=6):
        optimB = torch.optim.Adam(self.buyNN.parameters(), lr=self.lambd)
        optimS = torch.optim.Adam(self.sellNN.parameters(), lr=self.lambd)
        data_size = len(data)
        print("data size", data_size)
        input_size = self.input_size
        self.buyNN.train()
        self.sellNN.train()
        self.buyNN_tgt.train()
        self.sellNN_tgt.train()

        for k in range(epochs):
            lossB = torch.tensor(0, device=self.device)
            lossB2 = torch.tensor(0, device=self.device)
            lossS = torch.tensor(0, device=self.device)
            for i in range(data_size-input_size-1):
                states = data[i:i+input_size] #data[:,i:i+100] 
                #qt = buyNN(states)[:,1]
                actions = self.actBuy(states)
                buyPrice = torch.tensor(-1)
                sellPrice = torch.tensor(-1)
                if actions == 1: #buy
                    buyPrice = data[i+input_size] #data[:, i+100]
                    sold = False
                    j = 1
                    while not sold and i+input_size+j+1 < data_size:
                        statesSell = data[i+j:i+input_size+j] #data[:,i+j:i+100+j]
                        actions = self.actSell(statesSell)
                        if actions == 0: #hold
                            
                            reward = 100*(data[i+input_size+j+1]-data[i+input_size+j])/torch.abs(data[i+input_size+j])
                            #reward = data[i+input_size+j+1]-data[i+input_size+j]
                            nextStates = data[i+j+1:i+input_size+j+1] 
                            q_utils.store_transition(self.replayS, statesSell.cpu(), 0, reward.cpu(), nextStates.cpu())
                            if self.replayS.store_counter >= self.replay_capacity and self.replayB.store_counter >= self.replay_capacity:
                                #lrnS = self.learnS(optimS)
                                lossS = lossS + self.learnS(optimS)
                                #lossS = lossS + lrnS
                                #print("lrnS", lrnS)

                                if self.epsilonS > self.epsilon_min:
                                    self.epsilonS = self.epsilonS - self.epsilon_decay
                                self.parameter_updatesS += 1
                                if self.parameter_updatesS % self.tgt_net_update_freqS == 0:
                                    self.sellNN_tgt.load_state_dict(self.sellNN.state_dict())    
                                    
                            
                        else: #sell
                            sold = True
                            sellPrice = data[i+input_size+j]
                            
                        j+=1 

                    if sellPrice != -1: #=stock was sold
                        reward = 100*(sellPrice-buyPrice)/torch.abs(buyPrice)
                        #reward = sellPrice-buyPrice
                        nextStates = data[i+1:i+input_size+1]
                        q_utils.store_transition(self.replayB, states.cpu(), 1, reward.cpu(), nextStates.cpu())
                    
                    if self.replayS.store_counter >= self.replay_capacity and self.replayB.store_counter >= self.replay_capacity:
                        lossB = lossB + self.learnB(optimB)
                        self.parameter_updatesB += 1
                        if self.parameter_updatesB % self.tgt_net_update_freqB == 0:
                            self.buyNN_tgt.load_state_dict(self.buyNN.state_dict())
                        
                else:
                    reward = 100*(data[i+input_size]-data[i+input_size+1])/torch.abs(data[i+input_size])
                    nextStates = data[i+1:i+input_size+1]
                    #reward = data[i+input_size]-data[i+input_size+1]
                    
                    q_utils.store_transition(self.replayB, states.cpu(), 0, reward.cpu(), nextStates.cpu())
                    if self.replayS.store_counter >= self.replay_capacity and self.replayB.store_counter >= self.replay_capacity:
                        lossB = lossB + self.learnB(optimB)
                        self.parameter_updatesB += 1
                        if self.parameter_updatesB % self.tgt_net_update_freqB == 0:
                            self.buyNN_tgt.load_state_dict(self.buyNN.state_dict())
                if self.epsilonB > self.epsilon_min:
                    self.epsilonB = self.epsilonB - self.epsilon_decay   
                    
                    
                    
                    
                    
            print("epoch", k, "lossS", lossS.data/data_size, "lossB", lossB.data/data_size, "lossB2", lossB2.data/data_size)#, "gradient", gradient)  