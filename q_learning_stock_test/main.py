import agent
import q_utils
import test
import torch
import matplotlib.pyplot as plt

device = torch.device('cpu')
data = q_utils.get_data('SPY').squeeze().clone().detach().float().to(device)
#print("len", len(data))
data = data[:4000].to(device)
#print("mean", data.mean(), "var", data.var())
plt.plot(data.cpu().data)
plt.show

agnt = agent.Agent(replay_capacity=len(data))
agnt.learn(data, epochs=3)

test.test_net(agnt.buyNN, agnt.sellNN, 'fcel', agnt.input_size)