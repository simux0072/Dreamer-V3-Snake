import torch
import numpy
import sym_networks

symlog_func = lambda x: torch.sign(x) * torch.log(torch.abs(x) + 1)
symexp_func = lambda x: torch.sign(x) * (torch.exp(x) + 1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sym_net = sym_networks.sym_net().to(device)
optimizer = torch.optim.Adam(params=sym_net.parameters(), lr=0.0005)
mse_loss = torch.nn.MSELoss()

prev_model = torch.load("./models/sym_func_11720000.mdl")
sym_net.state_dict(prev_model['model_state_dict'])
optimizer.state_dict(prev_model['optimizer_state_dict'])

iter = prev_model['iter']

running_symlog_error = 0
running_symexp_error = 0

while True:
    iter += 1
    symlog_values = torch.from_numpy(numpy.random.uniform(low=-1000, high=1000, size=(200000, 1)).astype(numpy.float32)).to(device)
    symexp_values = torch.from_numpy(numpy.random.uniform(low=-10, high=10, size=(200000, 1)).astype(numpy.float32)).to(device)
    symlog_true = symlog_func(symlog_values)
    symexp_true = symexp_func(symexp_values)
    
    symlog_predict = sym_net.forward_symlog(symlog_values)
    symexp_predict = sym_net.forward_symexp(symexp_values)
    
    symlog_error: torch.Tensor = mse_loss(symlog_predict, symlog_true)
    symexp_error: torch.Tensor = mse_loss(symexp_predict, symexp_true)
    
    running_symlog_error += symlog_error.item()
    running_symexp_error += symexp_error.item()
    
    symlog_error.backward()
    symexp_error.backward()
    
    optimizer.step()
    optimizer.zero_grad()
    
    print(f"Iteration: {iter}", end='\r', flush=True)
    
    if iter % 5000 == 0:
        print(f"Iteration {iter}: Symlog error - {running_symlog_error / 1000}; Symexp error - {running_symexp_error / 1000}\n")
        running_symlog_error = 0
        running_symexp_error = 0
        
        if iter % 20000 == 0:
            torch.save({
                'iter': iter, 
                'model_state_dict': sym_net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
                }, f"./models/sym_func_{iter}.mdl")