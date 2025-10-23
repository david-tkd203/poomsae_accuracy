import numpy as np
from ..utils.geometry import angle_deg

def angles_from_landmarks(lmks, triplets):
    """
    lmks: lista de (x,y[,z,vis]) normalizados [0..1] o píxeles.
    triplets: dict nombre -> [i,j,k]
    """
    res = {}
    for name, (i,j,k) in triplets.items():
        if i>=len(lmks) or j>=len(lmks) or k>=len(lmks): 
            res[name] = np.nan; continue
        a = lmks[i][:2]; b = lmks[j][:2]; c = lmks[k][:2]
        res[name] = angle_deg(a,b,c)
    return res
