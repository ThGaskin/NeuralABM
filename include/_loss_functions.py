import torch

# Pytorch loss functions
LOSS_FUNCTIONS = {
    "l1loss": torch.nn.L1Loss,
    "mseloss": torch.nn.MSELoss,
    "crossentropyloss": torch.nn.CrossEntropyLoss,
    "ctcloss": torch.nn.CTCLoss,
    "nllloss": torch.nn.NLLLoss,
    "poissonnllloss": torch.nn.PoissonNLLLoss,
    "gaussiannllloss": torch.nn.GaussianNLLLoss,
    "kldivloss": torch.nn.KLDivLoss,
    "bceloss": torch.nn.BCELoss,
    "bcewithlogitsloss": torch.nn.BCEWithLogitsLoss,
    "marginrankingloss": torch.nn.MarginRankingLoss,
    "hingeembeddingloss": torch.nn.HingeEmbeddingLoss,
    "multilabelmarginloss": torch.nn.MultiLabelMarginLoss,
    "huberloss": torch.nn.HuberLoss,
    "smoothl1loss": torch.nn.SmoothL1Loss,
    "softmarginloss": torch.nn.SoftMarginLoss,
    "multilabelsoftmarginloss": torch.nn.MultiLabelSoftMarginLoss,
    "cosineembeddingloss": torch.nn.CosineEmbeddingLoss,
    "multimarginloss": torch.nn.MultiMarginLoss,
    "tripletmarginloss": torch.nn.TripletMarginLoss,
    "tripletmarginwithdistanceloss": torch.nn.TripletMarginWithDistanceLoss,
}
