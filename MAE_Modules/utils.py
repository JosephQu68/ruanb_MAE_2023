from Bio import pairwise2
import numpy as np
import torch
import random

def similarity_score(seq1, seq2):
    alignments = pairwise2.align.globalxx(seq1,seq2)
    lens = max(len(seq1), len(seq2))
    try:
        return  -alignments[0].score/lens
    except Exception as e:
        print(e)
        print(alignments)
        print(seq1)
        print(seq2)
        return 0
    
def estimate_gradient(z, q, beta, criterion, sigma=100):
    z_dim = z.shape[1:]
    u = np.random.normal(0, sigma, size=(q, z_dim[0],z_dim[1])).astype('float32')
    u = torch.from_numpy(u / np.linalg.norm(u, axis=1, keepdims=True)).to(device='cuda')

    f_0 = criterion(z)
    f_tmp = criterion(z + beta*u)
    print('Loss now: %f'%(f_0[0]))
    # print(f_0)
    u = u.to(device='cpu')
    # print(u.device)
    return torch.mean(z_dim[1] * u * np.expand_dims(np.expand_dims(f_tmp - f_0, 1),1)/ beta, dim=0,
                      keepdims=True).to(dtype = z.dtype, device = 'cuda')

def model_encode(seq):
    mask_rate = 0.15
    len_seq = len(seq)
    mask_len = int(len_seq * mask_rate)
    mask_idx = random.sample(range(len_seq), mask_len)
    
    raw_input = list(seq)
    for i,idx in enumerate(mask_idx):
        raw_input[idx] = '<extra_id_%d>'%i
    raw_input = [' '.join(raw_input)]
    inputs = tokenizer(raw_input, return_tensors='pt')['input_ids'].to('cuda')
    r1 = model.encoder.forward(inputs)['last_hidden_state']
    return r1

def model_decode(emb):
    outputs = torch.zeros((emb.shape[0],1),dtype = int).cuda()
    max_len = int(emb.shape[1] * 1.2)
    
    split_num = emb.shape[0] // INFER_BATCH_SIZE if emb.shape[0] > INFER_BATCH_SIZE else 1
    outputs = list(torch.chunk(outputs, split_num, dim = 0))
    emb = list(torch.chunk(emb, split_num, dim = 0))
    for i in range(0, max_len):
        for j in range(0, len(outputs)):
            out = model.decoder.forward(outputs[j], encoder_hidden_states= emb[j])
            out = model.lm_head(out['last_hidden_state'])
            out = torch.softmax(out, dim = -1)
            out = torch.argmax(out, dim = -1)
            outputs[j] = torch.cat((outputs[j], out[:,-1].unsqueeze(-1)),1)
    outputs = torch.concat(outputs, dim = 0)

    seq = []
    for single in outputs:
        end_idx = (torch.where(single == 1))
        if len(end_idx) == 0:
            end_idx = len(single)
        else:
            end_idx = end_idx[0][0]
        res = tokenizer.decode(single[1:end_idx])
        seq.append(res)
    
    seq = [''.join(single.split()) for single in seq]
    return seq

def loss_function(z, model,origin_seq ,weight=1, score=None, constraints=[],
                  weight_constraint=False):

    res = []
    num = z.shape[0]
    seq = model_decode(z)
    for i,strseq in enumerate(seq):
#         seq = model_decode(torch.unsqueeze(z[i], dim=0))
        
        loss_property = score(strseq, origin_seq) if score else 0

        loss_constraint = 0
        for c in constraints:
            loss_constraint += c(strseq)

        loss =  (loss_property + loss_constraint*weight if weight_constraint else
            loss_property*weight + loss_constraint)
        res.append(loss)
    return np.array(res)

def optimize(model, seq, q=100, base_lr=0.1, max_iter=1000, num_restarts=1,
             weight=0.1, beta=1, use_adam=False, early_stop=False, score=similarity_score,
             constraints=[amp_feature, toxic_feature], writer=None, run_str=None, results_dir='results',
             init_best={}, write_log=None, flip_weight=False):
    z_0 = model_encode(seq)     #获得序列的embedding
    print(seq)
    loss = partial(loss_function, model=model, origin_seq = seq,weight=weight, score=score,
                   constraints=[amp_feature, toxic_feature], weight_constraint=flip_weight)
    best = {'score': -np.inf, 'found': False, 'early_stop': False}
    best.update(init_best)
    # start_time = time()
    for k in range(num_restarts):
        if best['early_stop']:
            break
        z = z_0.clone()
        # traj_z, traj_loss = [z.clone().numpy()], [loss(z)]
        # adam = torch.optim.Adam([z], lr=base_lr)
        for i in (range(max_iter)):
            print('start itr %d'%i)
            # print('Begin calculating gradient')
            grad = estimate_gradient(z, q, beta, loss)  # 使用QMO计算离散梯度
            # print('Begin optim')
            if use_adam:
                z.grad = grad
                # adam.step() # 更新梯度
            else:
                lr = ((1 - i/max_iter)**0.5) * base_lr
                z -= grad * lr
            # z.clamp_(-1, 1)

            # traj_z.append(z.clone().numpy())
            # traj_loss.append(loss(z))

            # print('After one optim')
            mol = model_decode(z)[0]   # 将优化后的embedding还原为序列
            print('After optim ', mol)
            mol_score = score and -score(mol, seq)
            print('score is %f'%mol_score)
            print('AMP : %d, Toxic: %d'%(constraints[0](mol), constraints[1](mol)))
            # desc = write_log(writer, z, grad, z_0, mol[0], seq[0], i)
            if (score is None or mol_score > best['score']) and all(c(mol) == 0 for c in constraints):
                # best.update(desc)
                print('Bingo!')
                best.update(dict(step=i, z=z, z_0=z_0, seq=mol,
                                 score=mol_score, found=True, run=k,
                                  early_stop=early_stop))

                print(f'PASSED!')
                # np.savez(os.path.join(results_dir, run_str), **best)

                if early_stop:
                    break
            print()
        # np.savez(os.path.join(results_dir, f"TRAJ{k}_"+run_str),
        #             z=np.stack(traj_z), loss=np.array(traj_loss))
    if not best['found']:
        print('Search failed!')
    return best
