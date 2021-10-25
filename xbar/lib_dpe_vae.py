import sys
sys.path.append('..')
from lib_dpe_utils import DPE
import numpy as np

def vec_pn(Vec):
    # Make sure the vector is normalized
    if len(Vec.shape) == 1:
        Vec = np.expand_dims(Vec, 1)

    Vec_neg = np.zeros(Vec.shape)
    Vec_pos = np.zeros(Vec.shape)
    for i, vec in enumerate(Vec.T):
        vec = vec.reshape(-1)

        vec_pos = vec.copy()
        vec_pos[vec_pos < 0] = 0

        vec_neg = vec.copy()
        vec_neg[vec_neg > 0] = 0

        Vec_neg[:, i] = -vec_neg
        Vec_pos[:, i] = vec_pos
    return Vec_neg, Vec_pos

def dpe_pn(dpe, Vec, array, c_sel, **kwargs):
    """
    For input which has negative values
    """
    Vread = kwargs['Vread'] if 'Vread' in kwargs.keys() else 0.2
    tdly = kwargs['tdly'] if 'tdly' in kwargs.keys() else 1000
    
    Vec_neg, Vec_pos = vec_pn(Vec)

    Ipos = dpe.multiply(array,
                        Vec_pos,
                        c_sel=c_sel,
                        r_start=0, mode=0, Tdly=tdly)

    Ineg = dpe.multiply(array,
                        Vec_neg,
                        c_sel=c_sel,
                        r_start=0, mode=0, Tdly=tdly)

    Ires = Ipos-Ineg

    return Ires

def rram_sample(dpe, array, sample_range, targetG, **kwargs):
    """
    Sample gaussian distribution from rram crossbar array
    """
    
    vSetRamp = kwargs['vSetRamp'] if 'vSetRamp' in kwargs.keys() else [1.0, 2.5, 0.1]
    vGateSetRamp = kwargs['vGateSetRamp'] if 'vGateSetRamp' in kwargs.keys() else [1.0, 2.0, 0.1]
    vResetRamp = kwargs['vResetRamp'] if 'vResetRamp' in kwargs.keys() else [0.5, 3.5, 0.05]
    vGateResetRamp = kwargs['vGateResetRamp'] if 'vGateResetRamp' in kwargs.keys() else [5.0, 5.5, 0.1]
        
    maxSteps = kwargs['maxSteps'] if 'maxSteps' in kwargs.keys() else 100

    Gtol = kwargs['Gtol'] if 'Gtol' in kwargs.keys() else 10e-6
    Gtol_in = kwargs['Gtol_in'] if 'Gtol_in' in kwargs.keys() else Gtol
    Gtol_out = kwargs['Gtol_out'] if 'Gtol_out' in kwargs.keys() else Gtol

    #Msel = kwargs['Msel'] if 'Msel' in kwargs.keys() else np.ones(self.shape)

    saveHistory = kwargs['saveHistory'] if 'saveHistory' in kwargs.keys() else False
    maxRetry = kwargs['maxRetry'] if 'maxRetry' in kwargs.keys() else 5

    Tdly = kwargs['Tdly'] if 'Tdly' in kwargs.keys() else 1000
    method = kwargs['method'] if 'method' in kwargs.keys() else 'slow'

    Twidth = kwargs['Twidth'] if 'Twidth' in kwargs.keys() else 1000e-6
    TwidthSet = kwargs['TwidthSet'] if 'TwidthSet' in kwargs.keys() else Twidth
    TwidthReset = kwargs['TwidthReset'] if 'TwidthReset' in kwargs.keys() else Twidth
    
    stor_vec = np.ones((64, 64)) * targetG
    Msel = np.zeros((64, 64))
    Msel[sample_range[0]:sample_range[1], sample_range[2]:sample_range[3]] = 1

    dpe.tune_conductance(array,stor_vec,Gtol_in=Gtol_in,Gtol_out=Gtol_out,Msel=Msel,method='slow',maxSteps=maxSteps,Tdly=Tdly,maxRetry=maxRetry,TwidthReset=Twidth,TwidthSet=Twidth,vSetRamp=vSetRamp,vResetRamp=vResetRamp,vGateSetRamp=vGateSetRamp,vGateResetRamp=vGateResetRamp)
    
    # read conductance
    gmap = dpe.read(array, method='slow')[sample_range[0]:sample_range[1], sample_range[2]:sample_range[3]]
    # transfer to gaussian variable
    gaussian_sample = (gmap - targetG) / Gtol_in
    
    return gaussian_sample, gmap

def rram_stor(dpe, array, stor_range, pos, latent, **kwargs):
    g_ratio = kwargs['g_ratio'] if 'g_ratio' in kwargs.keys() else 50e-6
    vSetRamp = kwargs['vSetRamp'] if 'vSetRamp' in kwargs.keys() else [1.0, 2.5, 0.1]
    vGateSetRamp = kwargs['vGateSetRamp'] if 'vGateSetRamp' in kwargs.keys() else [1.0, 2.0, 0.1]
    vResetRamp = kwargs['vResetRamp'] if 'vResetRamp' in kwargs.keys() else [0.5, 3.5, 0.05]
    vGateResetRamp = kwargs['vGateResetRamp'] if 'vGateResetRamp' in kwargs.keys() else [5.0, 5.5, 0.1]
        
    maxSteps = kwargs['maxSteps'] if 'maxSteps' in kwargs.keys() else 80

    Gtol = kwargs['Gtol'] if 'Gtol' in kwargs.keys() else 5e-6
    Gtol_in = kwargs['Gtol_in'] if 'Gtol_in' in kwargs.keys() else Gtol
    Gtol_out = kwargs['Gtol_out'] if 'Gtol_out' in kwargs.keys() else Gtol

    #Msel = kwargs['Msel'] if 'Msel' in kwargs.keys() else np.ones(self.shape)

    saveHistory = kwargs['saveHistory'] if 'saveHistory' in kwargs.keys() else False
    maxRetry = kwargs['maxRetry'] if 'maxRetry' in kwargs.keys() else 5

    Tdly = kwargs['Tdly'] if 'Tdly' in kwargs.keys() else 1000
    method = kwargs['method'] if 'method' in kwargs.keys() else 'slow'

    Twidth = kwargs['Twidth'] if 'Twidth' in kwargs.keys() else 1000e-6
    TwidthSet = kwargs['TwidthSet'] if 'TwidthSet' in kwargs.keys() else Twidth
    TwidthReset = kwargs['TwidthReset'] if 'TwidthReset' in kwargs.keys() else Twidth
    assert stor_range[0] * stor_range[1] == len(latent.reshape(-1)), "dimension doesn't match!"
    assert stor_range[1] *2 + pos[1] < 64, "range exceed!"
    
    stor_vec = latent.reshape(stor_range[0], stor_range[1]) * g_ratio
    
    G = np.zeros((stor_vec.shape[0], stor_vec.shape[1] * 2))
    Gpos = stor_vec.copy()
    Gpos[Gpos < 0] = 0
    G[:, ::2] = Gpos
    Gneg = stor_vec.copy()
    Gneg[Gneg > 0] = 0
    G[:, 1::2] = -Gneg
    
    targetG = np.zeros((64, 64))
    Msel = np.zeros((64, 64))
    targetG[pos[0]:pos[0]+G.shape[0], pos[1]:pos[1]+G.shape[1]] = G
    Msel[pos[0]:pos[0]+G.shape[0], pos[1]:pos[1]+G.shape[1]] = 1
    
    dpe.tune_conductance(array,targetG,Gtol_in=Gtol_in,Gtol_out=8e-6,Msel=Msel,method='slow',maxSteps=maxSteps,Tdly=Tdly,maxRetry=maxRetry,TwidthReset=Twidth,TwidthSet=Twidth,vSetRamp=vSetRamp,vResetRamp=vResetRamp,vGateSetRamp=vGateSetRamp,vGateResetRamp=vGateResetRamp)
    
    g_latent_map = dpe.read(array, method='slow')[pos[0]:pos[0]+G.shape[0], pos[1]:pos[1]+G.shape[1]]
    g_reconsample = g_latent_map[:, ::2] - g_latent_map[:, 1::2]
    
    return g_reconsample.reshape(latent.shape[0], latent.shape[1]) / g_ratio, g_latent_map
    

def vae_mm(dpe, array, vec, geff, start_pos, lincor=True, **kwargs):
    """
    program conductance and do the matrix multiplications
    """
    vSetRamp = kwargs['vSetRamp'] if 'vSetRamp' in kwargs.keys() else [1.0, 2.5, 0.1]
    vGateSetRamp = kwargs['vGateSetRamp'] if 'vGateSetRamp' in kwargs.keys() else [1.0, 2.0, 0.1]
    vResetRamp = kwargs['vResetRamp'] if 'vResetRamp' in kwargs.keys() else [0.5, 3.5, 0.05]
    vGateResetRamp = kwargs['vGateResetRamp'] if 'vGateResetRamp' in kwargs.keys() else [5.0, 5.5, 0.1]
        
    maxSteps = kwargs['maxSteps'] if 'maxSteps' in kwargs.keys() else 10

    Gtol = kwargs['Gtol'] if 'Gtol' in kwargs.keys() else 5e-6
    Gtol_in = kwargs['Gtol_in'] if 'Gtol_in' in kwargs.keys() else Gtol
    Gtol_out = kwargs['Gtol_out'] if 'Gtol_out' in kwargs.keys() else Gtol

    #Msel = kwargs['Msel'] if 'Msel' in kwargs.keys() else np.ones(self.shape)

    saveHistory = kwargs['saveHistory'] if 'saveHistory' in kwargs.keys() else False
    maxRetry = kwargs['maxRetry'] if 'maxRetry' in kwargs.keys() else 5

    Tdly = kwargs['Tdly'] if 'Tdly' in kwargs.keys() else 1000
    method = kwargs['method'] if 'method' in kwargs.keys() else 'slow'

    Twidth = kwargs['Twidth'] if 'Twidth' in kwargs.keys() else 1000e-6
    TwidthSet = kwargs['TwidthSet'] if 'TwidthSet' in kwargs.keys() else Twidth
    TwidthReset = kwargs['TwidthReset'] if 'TwidthReset' in kwargs.keys() else Twidth
    
    Vread = kwargs['Vread'] if 'Vread' in kwargs.keys() else 0.2
    tdly = kwargs['tdly'] if 'tdly' in kwargs.keys() else 500
    
    assert vec.shape[0] == geff.shape[0], "dimension doesn't match!"
    assert geff.shape[0] + start_pos[0] <= 64, "mapping exceed number of rows!"
    assert geff.shape[1] + start_pos[1] <= 64, "mapping exceed number of columns!"
    
    stor_vec = np.zeros((64, 64))
    stor_vec[start_pos[0]:start_pos[0] + geff.shape[0], start_pos[1]:start_pos[1] + geff.shape[1]] = geff
    Msel = np.zeros((64, 64))
    #Msel = np.ones((64, 64))
    Msel[start_pos[0]:start_pos[0] + geff.shape[0], start_pos[1]:start_pos[1] + geff.shape[1]] = 1

    dpe.tune_conductance(array,stor_vec,Gtol_in=Gtol,Gtol_out=8e-6,Msel=Msel,method='slow',maxSteps=maxSteps,Tdly=Tdly,maxRetry=maxRetry,TwidthReset=Twidth,TwidthSet=Twidth,vSetRamp=vSetRamp,vResetRamp=vResetRamp,vGateSetRamp=vGateSetRamp,vGateResetRamp=vGateResetRamp)
        
    # read_conductance
    gmap = dpe.read(array, method='slow')
    #transfer input to crossbar input, make sure xbarinput.shape[0] == 64
    if vec.shape[0] == 64:
        xbarinput = vec
    else:
        xbarinput = np.zeros((64, vec.shape[1]))
        xbarinput[start_pos[0]:start_pos[0] + vec.shape[0], :] = vec
        
    # linear correction
    if lincor:
        if xbarinput.min() < 0:
            lin_cor = np.zeros((geff.shape[1],2))
            #rand_in = np.random.rand(vec.shape[0], 500)
            #testin = np.zeros((64, 500))
            #testin[start_pos[0]:start_pos[0] + vec.shape[0], :] = rand_in
            testin = xbarinput[:,:100]
            results_cali = (testin.T@gmap).T
            results = dpe_pn(dpe, testin, array, [start_pos[1], start_pos[1] + geff.shape[1]], tdly=tdly).T
            for i in range(results.shape[0]):
                p1=np.polyfit(results[i],results_cali[i + start_pos[1]],1)
                lin_cor[i]=p1
        else:
            lin_cor = np.zeros((geff.shape[1],2))
            #rand_in = np.random.rand(vec.shape[0], 500)
            #testin = np.zeros((64, 500))
            #testin[start_pos[0]:start_pos[0] + vec.shape[0], :] = rand_in
            testin = xbarinput[:,:100]
            results_cali = (testin.T@gmap).T
            results = dpe.multiply(array, testin, c_sel=[start_pos[1], start_pos[1] + geff.shape[1]], Tdly=tdly).T
            for i in range(results.shape[0]):
                p1=np.polyfit(results[i],results_cali[i + start_pos[1]],1)
                lin_cor[i]=p1
        
    # perform matrix multiplication
    if xbarinput.min() < 0:
        output = dpe_pn(dpe, xbarinput, array, [start_pos[1], start_pos[1] + geff.shape[1]], tdly=tdly)
    else:
        output = dpe.multiply(array, xbarinput, c_sel=[start_pos[1], start_pos[1] + geff.shape[1]], r_start=0, mode=0, Tdly=tdly)
        
    output = dpe.lin_corr(output, lin_cor)
        
        
    return output, gmap[start_pos[0]:start_pos[0] + geff.shape[0], start_pos[1]:start_pos[1] + geff.shape[1]]
    
    
    