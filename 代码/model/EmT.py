import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import math

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# ==================== 基础图卷积模块（保持不变） ====================

class GraphConvolution(Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        torch.nn.init.xavier_uniform_(self.weight, gain=1.414)
        if bias:
            self.bias = Parameter(torch.zeros((1, 1, out_features), dtype=torch.float32))
        else:
            self.register_parameter('bias', None)

    def forward(self, x, adj):
        output = torch.matmul(x, self.weight) - self.bias
        output = F.relu(torch.matmul(adj, output))
        return output


class GCN(Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GCN, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, data):
        graph, adj = data
        adj = self.norm_adj(adj)
        support = torch.matmul(graph, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            output = (F.relu(output + self.bias), adj)
        else:
            output = (F.relu(output), adj)
        return output

    def norm_adj(self, adj):
        rowsum = torch.sum(adj, dim=-1)
        mask = torch.zeros_like(rowsum)
        mask[rowsum == 0] = 1
        rowsum += mask
        d_inv_sqrt = torch.pow(rowsum, -0.5)
        d_mat_inv_sqrt = torch.diag_embed(d_inv_sqrt)
        adj = torch.mm(torch.mm(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)
        return adj


class ChebyNet(Module):
    def __init__(self, K, in_feature, out_feature):
        super(ChebyNet, self).__init__()
        self.K = K
        self.filter_weight, self.filter_bias = self.init_fliter(K, in_feature, out_feature)

    def init_fliter(self, K, feature, out, bias=True):
        weight = nn.Parameter(torch.FloatTensor(K, 1, feature, out), requires_grad=True)
        nn.init.normal_(weight, 0, 0.1)
        bias_ = None
        if bias:
            bias_ = nn.Parameter(torch.zeros((1, 1, out), dtype=torch.float32), requires_grad=True)
            nn.init.normal_(bias_, 0, 0.1)
        return weight, bias_

    def get_L(self, adj):
        degree = torch.sum(adj, dim=1)
        degree_norm = torch.div(1.0, torch.sqrt(degree) + 1.0e-5)
        degree_matrix = torch.diag(degree_norm)
        L = - torch.matmul(torch.matmul(degree_matrix, adj), degree_matrix)
        return L

    def chebyshev(self, x, L):
        x1 = torch.matmul(L, x)
        x_ = torch.stack((x, x1), dim=1)
        if self.K > 1:
            for k in range(2, self.K):
                x_current = 2 * torch.matmul(L, x_[:, -1]) - x_[:, -2]
                x_current = x_current.unsqueeze(dim=1)
                x_ = torch.cat((x_, x_current), dim=1)
        x_ = x_.permute(1, 0, 2, 3)
        out = torch.matmul(x_, self.filter_weight)
        out = torch.sum(out, dim=0)
        out = F.relu(out + self.filter_bias)
        return out

    def forward(self, data):
        x, adj = data
        L = self.get_L(adj)
        out = self.chebyshev(x, L)
        return out, adj


# ==================== Transformer基础子模块 ====================

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


# ==================== GraphEncoder ====================

class GraphEncoder(nn.Module):
    def __init__(self, num_layers, num_node, in_features, out_features, K,
                 graph2token='Linear', encoder_type='GCN'):
        super(GraphEncoder, self).__init__()
        self.graph2token = graph2token
        self.K = K
        assert graph2token in ['Linear', 'AvgPool', 'MaxPool', 'Flatten'], "graph2vector type is not supported!"
        if graph2token == 'Linear':
            self.tokenizer = nn.Linear(num_node * out_features, out_features)
        else:
            self.tokenizer = None

        layers = []
        for i in range(num_layers):
            if i == 0:
                layer = self.get_layer(encoder_type, in_features, out_features)
            else:
                layer = self.get_layer(encoder_type, out_features, out_features)
            layers.append(layer)
        self.encoder = nn.Sequential(*layers)

    def get_layer(self, encoder_type, in_features, out_features):
        assert encoder_type in ['Cheby', 'GCN'], "encoder type is not supported!"
        if encoder_type == 'GCN':
            return GCN(in_features, out_features)
        if encoder_type == 'Cheby':
            return ChebyNet(self.K, in_features, out_features)

    def forward(self, x, adj):
        output = self.encoder((x, adj))
        x, _ = output
        if self.tokenizer is not None:
            x = x.view(x.size(0), -1)
            output = self.tokenizer(x)
        else:
            if self.graph2token == 'AvgPool':
                output = torch.mean(x, dim=-1)
            elif self.graph2token == 'MaxPool':
                output = torch.max(x, dim=-1)[0]
            else:
                output = x.view(x.size(0), -1)
        return output


# ==================== GRU 时序混合器（层数可配） ====================

class Attention(nn.Module):
    """
    TCT-Regr 的时序混合器
    - GRU 层数由参数控制，默认 1 层双向
    """
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., gru_layers=1):
        super().__init__()
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.dim_head = dim_head

        self.to_qkv = nn.Linear(dim, dim_head, bias=False)

        # ===== GRU 层数由参数控制 =====
        self.token_mixer = nn.GRU(
            input_size=dim_head,
            hidden_size=dim_head,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.0 if gru_layers == 1 else 0.3
        )

        out_dim = 2 * dim_head
        self.to_out = nn.Sequential(
            nn.Linear(out_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        out = self.to_qkv(x)
        out, _ = self.token_mixer(out)
        return self.to_out(out)


# ==================== TTransformer ====================

class TTransformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0., gru_layers=1):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head,
                                       dropout=dropout, gru_layers=gru_layers)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


# ==================== 双流 EmT + 可学习/固定 α 门控融合 ====================

class EmT(nn.Module):
    """
    双流 EmT：
    - rPSD 和 DE 各自由独立的 RMPG 编码（参数不共享）
    - 门控标量 α 可选择可学习或固定
    - GRU 层数可配置，用于消融
    - 单流模式（single_stream）支持 rPSD 或 DE 走对应分支
    """
    def __init__(self,
                 layers_graph=[1, 2],
                 layers_transformer=2,
                 num_adj=2,
                 num_chan=32,
                 num_feature=6,
                 hidden_graph=32,
                 K=3,
                 num_head=32,
                 dim_head=32,
                 dropout=0.5,
                 num_class=1,
                 graph2token='Linear',
                 encoder_type='Cheby',
                 alpha_init=0.7,
                 single_stream='r',
                 fixed_alpha=None,
                 gru_layers=1):
        super(EmT, self).__init__()

        self.num_adj = num_adj
        self.num_chan = num_chan
        self.num_feature = num_feature
        self.hidden_graph = hidden_graph
        self.graph_encoder_type = encoder_type
        self.K = K
        self.single_stream = single_stream
        self.fixed_alpha = fixed_alpha
        self.gru_layers = gru_layers

        # ===== 双流独立 RMPG（参数不共享） =====
        self.r_GE1 = GraphEncoder(
            num_layers=layers_graph[0], num_node=num_chan,
            in_features=num_feature, out_features=hidden_graph,
            K=K, graph2token=graph2token, encoder_type=encoder_type
        )
        self.r_GE2 = GraphEncoder(
            num_layers=layers_graph[1], num_node=num_chan,
            in_features=num_feature, out_features=hidden_graph,
            K=K, graph2token=graph2token, encoder_type=encoder_type
        )

        self.d_GE1 = GraphEncoder(
            num_layers=layers_graph[0], num_node=num_chan,
            in_features=num_feature, out_features=hidden_graph,
            K=K, graph2token=graph2token, encoder_type=encoder_type
        )
        self.d_GE2 = GraphEncoder(
            num_layers=layers_graph[1], num_node=num_chan,
            in_features=num_feature, out_features=hidden_graph,
            K=K, graph2token=graph2token, encoder_type=encoder_type
        )

        # ===== 双流可学习邻接矩阵 =====
        self.r_adjs = nn.Parameter(torch.FloatTensor(num_adj, num_chan, num_chan), requires_grad=True)
        nn.init.xavier_uniform_(self.r_adjs)

        self.d_adjs = nn.Parameter(torch.FloatTensor(num_adj, num_chan, num_chan), requires_grad=True)
        nn.init.xavier_uniform_(self.d_adjs)

        # ===== 门控参数 α：可学习或固定 =====
        if fixed_alpha is not None:
            self.alpha_param = nn.Parameter(torch.tensor(float(fixed_alpha)), requires_grad=False)
            print(f"[EmT] alpha fixed at {fixed_alpha}")
        else:
            self.alpha_param = nn.Parameter(torch.tensor(alpha_init), requires_grad=True)
            print(f"[EmT] alpha learnable, init={alpha_init}")

        # ===== 残差旁路（双流各有一个） =====
        self.to_GNN_out_r = nn.Linear(num_chan * num_feature, hidden_graph, bias=False)
        self.to_GNN_out_d = nn.Linear(num_chan * num_feature, hidden_graph, bias=False)

        # ===== 时序 Transformer（GRU 层数可配） =====
        self.transformer = TTransformer(
            depth=layers_transformer,
            dim=hidden_graph,
            heads=num_head,
            dim_head=dim_head,
            dropout=dropout,
            mlp_dim=dim_head,
            gru_layers=gru_layers
        )

        # ===== 回归预测头 =====
        self.MLP = nn.Sequential(
            nn.Linear(hidden_graph, num_class)
        )

        # ===== 特征缓存（可视化） =====
        self.feature_before = None
        self.feature_after = None

    def get_adj(self, adjs, self_loop=False):
        num_nodes = adjs.shape[-1]
        adj = F.relu(adjs + adjs.transpose(-2, -1))
        if self_loop:
            adj = adj + torch.eye(num_nodes).to(DEVICE)
        return adj

    def _encode_single_stream(self, x, ge1, ge2, adjs, to_gnn_out):
        b, s, c, f = x.size()
        x_flat = rearrange(x, 'b s c f -> (b s) c f')

        if self.graph_encoder_type == 'Cheby':
            adj_list = [self.get_adj(adjs[i], self_loop=False) for i in range(adjs.size(0))]
        else:
            adj_list = [self.get_adj(adjs[i], self_loop=True) for i in range(adjs.size(0))]

        x_res = x_flat.contiguous().view(x_flat.size(0), -1)
        x_res = to_gnn_out(x_res)

        x1 = ge1(x_flat, adj_list[0])
        x2 = ge2(x_flat, adj_list[1])

        x_fused = torch.stack((x_res, x1, x2), dim=1)
        x_fused = torch.mean(x_fused, dim=1)

        token = rearrange(x_fused, '(b s) h -> b s h', b=b, s=s)
        return token

    def forward(self, rpsd=None, de=None, x=None):
        # ===== 单流模式 =====
        if x is not None and rpsd is None and de is None:
            if self.single_stream == 'd':
                token = self._encode_single_stream(
                    x, self.d_GE1, self.d_GE2, self.d_adjs, self.to_GNN_out_d
                )
            else:
                token = self._encode_single_stream(
                    x, self.r_GE1, self.r_GE2, self.r_adjs, self.to_GNN_out_r
                )

            self.feature_before = token
            token = self.transformer(token)
            self.feature_after = token
            return self.MLP(token)

        # ===== 双流模式 =====
        if rpsd is None or de is None:
            raise ValueError("EmT requires either 'x' or both 'rpsd' and 'de'.")

        token_r = self._encode_single_stream(rpsd, self.r_GE1, self.r_GE2, self.r_adjs, self.to_GNN_out_r)
        token_d = self._encode_single_stream(de, self.d_GE1, self.d_GE2, self.d_adjs, self.to_GNN_out_d)

        # 门控融合：固定或可学习
        if self.fixed_alpha is not None:
            alpha = torch.tensor(float(self.fixed_alpha), device=self.alpha_param.device)
        else:
            alpha = torch.sigmoid(self.alpha_param)

        fused = alpha * token_r + (1 - alpha) * token_d

        self.feature_before = fused
        fused = self.transformer(fused)
        self.feature_after = fused

        out = self.MLP(fused)
        return out


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    data_r = torch.ones((2, 96, 32, 6))
    data_d = torch.ones((2, 96, 32, 6))

    model = EmT(
        layers_graph=[1, 2], layers_transformer=2, num_adj=2,
        num_chan=32, num_feature=6, hidden_graph=32, K=3,
        num_head=32, dim_head=32, dropout=0.5, num_class=1,
        graph2token='Linear', encoder_type='Cheby', alpha_init=0.7,
        single_stream='r', fixed_alpha=None, gru_layers=1
    )
    print(f"Total parameters: {count_parameters(model)}")
    out = model(rpsd=data_r, de=data_d)
    print(f"Output shape: {out.shape}")

    # 测试固定 α
    model_fa = EmT(
        layers_graph=[1, 2], layers_transformer=2, num_adj=2,
        num_chan=32, num_feature=6, hidden_graph=32, K=3,
        num_head=32, dim_head=32, dropout=0.5, num_class=1,
        graph2token='Linear', encoder_type='Cheby', alpha_init=0.7,
        single_stream='r', fixed_alpha=0.5, gru_layers=1
    )
    out_fa = model_fa(rpsd=data_r, de=data_d)
    print(f"Fixed-alpha output shape: {out_fa.shape}")

    # 测试 2 层 GRU
    model_gru2 = EmT(
        layers_graph=[1, 2], layers_transformer=2, num_adj=2,
        num_chan=32, num_feature=6, hidden_graph=32, K=3,
        num_head=32, dim_head=32, dropout=0.5, num_class=1,
        graph2token='Linear', encoder_type='Cheby', alpha_init=0.7,
        single_stream='r', fixed_alpha=None, gru_layers=2
    )
    out_gru2 = model_gru2(rpsd=data_r, de=data_d)
    print(f"GRU-2 output shape: {out_gru2.shape}")