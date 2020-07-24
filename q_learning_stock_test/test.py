import matplotlib.pyplot as plt
import q_utils
import torch

def test_net(buyNN, sellNN, symbol, input_size, device=torch.device('cpu')):
    dataTest = q_utils.get_data(symbol).squeeze().clone().detach().float().to(device)

    plt_data = dataTest#[4000:5000]
    plt_size = len(plt_data)
    #plt_data = plt_data - plt_data.mean()
    #plt_data = plt_data / plt_data.std()
    money = 10000
    buyAmount = 10000
    holdingAmount = 0

    buyNN.eval()
    sellNN.eval()

    plt.plot(plt_data.cpu().data)
    holding = False
    for i in range(plt_size-input_size):
        states = plt_data[i:i+input_size]
        q = buyNN(states.unsqueeze(0))
        q2 = sellNN(states.unsqueeze(0))
        if holding != True:
            if torch.argmax(q) == 1:
                holding = True
                amount = int(buyAmount/plt_data[i+input_size])
                price = amount*plt_data[i+input_size]
                holdingAmount = amount
                money = money - price
                plt.plot(i+input_size, plt_data[i+input_size].cpu(), 'ro')
        else:
            if q2 < 0:
                holding = False
                money = holdingAmount * plt_data[i+input_size] + money
                holdingAmount = 0
                plt.plot(i+input_size, plt_data[i+input_size].cpu(), 'bo')
        #print("i", i+input_size)
        #print("q", q)
        #print("q2", q2)
        #print(money)
    print("final money:", money + holdingAmount * plt_data[plt_size-1])    
    print("stock grew to:", plt_data[plt_size-1]/plt_data[input_size]*buyAmount)
    plt.show()    