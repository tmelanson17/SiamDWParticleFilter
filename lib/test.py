import models.models as model
import utils.utils as utils

import cv2
import numpy as np
from torch.autograd import Variable

# Test script for running the model raw
if __name__ == "__main__":
    # Currently randomly initialized
    t = model.SiamFCRes22W()
    utils.load_pretrain(t, "/home/noisebridge/development/SiamDWParticleFilter/models/SiamFCRes22W.pth")
    # Collect template image
    template = cv2.imread("/home/noisebridge/development/SiamDWParticleFilter/cat1.jpeg")
    test_good = cv2.imread("/home/noisebridge/development/SiamDWParticleFilter/cat2.jpeg")
    test_bad = cv2.imread("/home/noisebridge/development/SiamDWParticleFilter/dog.jpeg")
    # template=np.ones([500,200,3])
    # template=np.random.random([500,200,3])
    template_torch = utils.im_to_torch(template)
    template_var = Variable(template_torch.unsqueeze(0))
    t.template(template_var) #template_var.cuda()
    # Test against random image
    # test=255*np.ones_like(template) # np.random.random([500,200,3])
    # test_var = Variable(utils.im_to_torch(test).unsqueeze(0))
    # output = t.track(test_var)
    # print(output)

    # Positive test
    test_good_var = Variable(utils.im_to_torch(test_good).unsqueeze(0))
    test_bad_var = Variable(utils.im_to_torch(test_bad).unsqueeze(0))
    output_bad = t.track(test_bad_var)
    output_good = t.track(test_good_var)
    print("Output good, bad (good should be > bad)")
    print(output_good)
    print(output_bad)
