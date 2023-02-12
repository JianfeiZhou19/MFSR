import time
import torch
from thop import profile
from sr_model.LF_InterNet import get_model

def cal_param(net):
    # net = net(5, 2).cuda()
    input = torch.randn(1, 1, 80, 80).cuda()
    total = sum([param.nelement() for param in net.parameters()])
    flops, params = profile(net, inputs=(input,))
    print('   Number of parameters: %.2fM' % (total / 1e6))
    print('   Number of FLOPs: %.2fG' % (flops / 1e9))


if __name__ == "__main__":
    x = torch.rand(1, 1, 80, 80).to("cuda:0")

    torch.cuda.set_device("cuda:0")
    net = get_model().to("cuda:0")
    start = time.time()
    for _ in range(10):
        out = net(x)
    end = time.time()

    total = sum([param.nelement() for param in net.parameters()])
    print('   Number of parameters: %.4fM' % (total / 1e6))
    print("   Spend time: ", (end-start)/10)
    cal_param(net=net)
