import torch
import esm
from tqdm import tqdm


class ESMEmbedder:
    def __init__(self, num_layers):
        self.num_layers = num_layers
        self.models = {
            6: "esm2_t6_8M_UR50D",
            12: "esm2_t12_35M_UR50D",
            30: "esm2_t30_150M_UR50D",
            33: "esm2_t33_650M_UR50D",
        }

        if self.num_layers not in self.models:
            raise ValueError(
                f"Unsupported number of layers: {self.num_layers}. Supported layers are: {list(self.models.keys())}"
            )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.alphabet = getattr(esm.pretrained, self.models[self.num_layers])()
        self.batch_converter = self.alphabet.get_batch_converter()
        self.model.eval().to(self.device)

    def run(self, data, layers=None, contacts=False):
        if layers is None:
            layers = range(self.num_layers + 1)
        results = []
        for prot in tqdm(data):
            batch_labels, batch_strs, batch_tokens = self.batch_converter([prot])
            batch_tokens = batch_tokens.to(self.device)  # Ensure tokens are on the GPU
            with torch.no_grad():
                i = self.model.forward(batch_tokens, repr_layers=layers, return_contacts=contacts)
                detached_i = {}
                for k, v in i.items():
                    if isinstance(v, dict):  # Check if value is a dictionary (like "representations")
                        detached_i[k] = {k1: v1.detach().cpu() for k1, v1 in v.items()}
                    else:
                        detached_i[k] = v.detach().cpu()
                results.append(detached_i)
        return results