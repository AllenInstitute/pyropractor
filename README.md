
# pyropractor
When you need to get your features into better alignment,
it's time to see the pyropractor


Installation instructions
=========================
WIP

Usage Example
=========================

In the example below we create two 100x2 feature arrays
representing 100 samples and 2 feature columns. Note, the
feature arrays need not, and typically won't, have the same number of samples.  


In this demonstration, we are happy with how well aligned 
one of the features is (this is our static feature) but we 
want to align the other(s) (tunable feature(s))
``` 
import numpy as np
import matplotlib.pyplot as plt
from pyropractor.chamfer import ChamferAlign

# define reference features
reference_tunable_feat = np.random.normal(0,0.5,100)
reference_tunable_feat[0:30] = abs(reference_tunable_feat[0:30])*10
reference_tunable_feat[80:] = abs(reference_tunable_feat[80:])*20
reference_static_feat = np.arange(0,1,1/len(reference_tunable_feat))

# introduce misalignment
misaligned_tunable_feat = (reference_tunable_feat*0.3)+0.113

# add a small amount of noise to our static feature
static_feat_noise = np.random.normal(0,0.005,len(reference_static_feat))
misaligned_ys = reference_static_feat+static_feat_noise

# create (100,2) matrices 
reference_X = np.stack((reference_tunable_feat,reference_static_feat)).T
misaligned_X = np.stack((misaligned_tunable_feat,misaligned_ys)).T

chmf = ChamferAlign(epochs=500,
                   subsample=1.0,
                   n_random_starts=5,
                   static_feature_index=1,
                   learning_rate=0.01
                   )
aligned_X = chmf.align(align_X=misaligned_X, reference_X=reference_X)

plt.scatter(reference_X[:,0],reference_X[:,1],label='reference',c='k',alpha=0.75)
plt.scatter(misaligned_X[:,0],misaligned_X[:,1],label='misaligned',c='firebrick',alpha=0.75)
plt.scatter(aligned_X[:,0],aligned_X[:,1],label='aligned',c='steelblue',alpha=0.75)
plt.xlabel("Tunable Feature")
plt.ylabel("Static Feature")
plt.title("Transforming Misaligned Data to A Reference PointCloud")
plt.legend()

# measure change in chamfer distance
chmf_dist_before = chmf.transform_data[0]['original_chamfer_distance']
chmf_dist_after = chmf.transform_data[0]['minimized_chamfer_distance']
print(f"Misaligned Chamfer-Distance: {chmf_dist_before}")
print(f"Aligned Chamfer-Distance: {chmf_dist_after}")

Misaligned Chamfer-Distance: 6.13359260559082
Aligned Chamfer-Distance: 0.3389962315559387

```
![Alt text](demo/AlignStaticDemo.png?raw=true "Title")




Statement of Support
====================
This code is an important part of the internal Allen Institute code base and we are actively using and maintaining it. Issues are encouraged, but because this tool is so central to our mission pull requests might not be accepted if they conflict with our existing plans.