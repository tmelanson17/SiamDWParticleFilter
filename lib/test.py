import models.models as model
import utils.utils as utils

import numpy as np
from torch.autograd import Variable

# Test script for running the model raw
if __name__ == "__main__":
    # Currently randomly initialized
    t = model.SiamFCRes22()
    # Collect template image
    template=np.ones([500,200,3])
    template_torch = utils.im_to_torch(template)
    template_var = Variable(template_torch.unsqueeze(0))
    t.template(template_var) #template_var.cuda()
    # Test against random image
    test=np.random.random([500,200,3])
    test_var = Variable(utils.im_to_torch(test).unsqueeze(0))
    t.track(test_var)
