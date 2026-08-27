import torch



class ReBrainFormer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args=args
        self.lambda1=args.lambda1
        self.lambda2=args.lambda2
        self.lambda3=args.lambda3
        self.Wq=nn.Parameter(args.Nodes, args.Nodes)
        self.Wv=nn.Parameter(args.Nodes, args.Nodes)
        self.Wk=nn.Parameter(args.Nodes, args.Nodes)
        self.classes=args.classes
        self.classifier=nn.Linear(args.Nodes, self.classes)
    def self_attention_map(self,H, K, N_E, C):
        Q=H@self.Wq
        K=H@self.Wk
        V=H@self.Wv
        attention_scores=((Q@K.T)/math.sqrt(self.args.Nodes))+
                            self.lambda1*K+
                            self.lambda2*N_E+
                            self.lambda3*C
        return nn.Softmax(attention_scores, dim=-1)@V
    

    def forward(self, x):
        H=x[:,0,:]
        K=x[:,1,:]
        NE=x[:,2,:]
        C=x[:,3,:]
        Emb=self.self_attention_map(H,K,NE,C)
        pooled=Emb.mean(dim=0)
        return self.classifier(pooled)

