import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from losses import hic_loss_fn, rna_loss_fn
from utils import reparameterize
from vmf import sample_vmf, vmf_kl_uniform


class HiCEncoder(nn.Module):
    def __init__(self, input_dim, hid_dim, latent_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hid_dim = hid_dim
        self.latent_dim = latent_dim

        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=2)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)

        dummy_input = torch.zeros(1, 1, self.input_dim, self.input_dim)
        self.conv_size, self.h = self.get_shape(dummy_input)

        self.fc = nn.Linear(self.conv_size, self.hid_dim)
        self.mu = nn.Linear(self.hid_dim, self.latent_dim)
        self.logvar = nn.Linear(self.hid_dim, self.latent_dim)

    def get_shape(self, dummy):
        out = self.conv1(dummy)
        out = self.pool1(out)
        out = self.conv2(out)
        out = self.pool2(out)
        _, _, h, _ = out.shape
        conv_size = int(np.prod(out.size()))
        return conv_size, h

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(-1, self.conv_size)
        x = F.relu(self.fc(x))
        return self.mu(x), self.logvar(x)


class HiCDecoder(nn.Module):
    def __init__(self, hid_dim, z_dim, conv_size, h):
        super().__init__()
        self.z_dim = z_dim
        self.hid_dim = hid_dim
        self.conv_size = conv_size
        self.h = h

        self.fc_dec = nn.Linear(self.z_dim, self.hid_dim)
        self.fc_dec_out = nn.Linear(self.hid_dim, self.conv_size)
        self.deconv1 = nn.ConvTranspose2d(16, 6, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(6, 1, kernel_size=5, stride=2, padding=2, output_padding=1)

    def forward(self, z):
        x = F.relu(self.fc_dec(z))
        x = F.relu(self.fc_dec_out(x))
        x = x.view(-1, 16, self.h, self.h)
        x = F.relu(self.deconv1(x))
        return self.deconv2(x)


class FCLayers(nn.Module):
    def __init__(
        self,
        n_in,
        n_out,
        n_layers,
        n_hidden=128,
        dropout_rate=0.1,
        use_batch_norm=True,
        use_activation=True,
    ):
        super().__init__()
        modules = []
        layers_dim = [n_in] + [n_hidden] * (n_layers - 1) + [n_out]

        for in_dim, out_dim in zip(layers_dim[:-1], layers_dim[1:]):
            layer_modules = [nn.Linear(in_dim, out_dim)]
            if use_batch_norm:
                layer_modules.append(nn.BatchNorm1d(out_dim, momentum=0.01, eps=0.001))
            if use_activation:
                layer_modules.append(nn.LeakyReLU())
            layer_modules.append(nn.Dropout(p=dropout_rate))
            modules.append(nn.Sequential(*layer_modules))

        self.layers = nn.Sequential(*modules)

    def forward(self, x):
        return self.layers(x)


class RNAEncoder(nn.Module):
    def __init__(
        self,
        n_input,
        n_fc_layers,
        n_latent,
        n_hidden=128,
        dropout_rate=0.1,
        use_batch_norm=True,
        use_activation=True,
    ):
        super().__init__()
        modules = []
        current_dim = n_input

        while current_dim > n_hidden:
            next_dim = max(n_hidden, current_dim // 10)
            modules.append(
                FCLayers(
                    n_in=current_dim,
                    n_out=next_dim,
                    n_layers=1,
                    n_hidden=next_dim,
                    dropout_rate=dropout_rate,
                    use_batch_norm=use_batch_norm,
                    use_activation=use_activation,
                )
            )
            current_dim = next_dim

        modules.append(
            FCLayers(
                n_in=n_hidden,
                n_out=n_hidden,
                n_layers=n_fc_layers,
                n_hidden=n_hidden,
                dropout_rate=dropout_rate,
                use_batch_norm=use_batch_norm,
                use_activation=use_activation,
            )
        )

        self.encoder = nn.Sequential(*modules)
        self.mu = nn.Linear(n_hidden, n_latent)
        self.logvar = nn.Linear(n_hidden, n_latent)

    def forward(self, x):
        y = self.encoder(x)
        return self.mu(y), self.logvar(y)


class RNADecoder(nn.Module):
    def __init__(
        self,
        n_output,
        n_fc_layers,
        z_dim,
        n_hidden=128,
        dropout_rate=0.1,
        use_batch_norm=True,
        use_activation=True,
    ):
        super().__init__()
        self.decoder = FCLayers(
            n_in=z_dim,
            n_out=n_hidden,
            n_layers=n_fc_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            use_batch_norm=use_batch_norm,
            use_activation=use_activation,
        )
        self.theta = nn.Linear(n_hidden, n_output)
        self.logit_pi = nn.Linear(n_hidden, n_output)
        self.ratio = nn.Sequential(
            nn.Linear(n_hidden, n_output),
            nn.Softmax(dim=-1),
        )

    def forward(self, log_l, z):
        result = self.decoder(z)

        theta_raw = self.theta(result)
        theta_raw = torch.nan_to_num(theta_raw, nan=0.0, posinf=10.0, neginf=-10.0)
        theta_raw = torch.clamp(theta_raw, min=-10.0, max=10.0)
        theta = F.softplus(theta_raw) + 1e-4
        theta = torch.clamp(theta, min=1e-4, max=1e4)

        logit_pi = self.logit_pi(result)
        ratio = self.ratio(result)
        lib = torch.exp(torch.clamp(log_l, -10.0, 10.0))
        rate = lib * ratio + 1e-8

        return theta, logit_pi, ratio, rate


class CombineEncoder(nn.Module):
    def __init__(
        self,
        rna_input_dim,
        rna_fc_layers,
        rna_hid_dim,
        chr_input_dims,
        hic_hid_dim,
        latent_dim,
        z_dim,
        mode,
    ):
        super().__init__()
        self.RNAencoder = RNAEncoder(
            n_input=rna_input_dim,
            n_fc_layers=rna_fc_layers,
            n_hidden=rna_hid_dim,
            n_latent=latent_dim,
        )
        self.n_chr = len(chr_input_dims)
        self.HiCencoders = nn.ModuleList(
            [
                HiCEncoder(
                    input_dim=chr_input_dims[i],
                    hid_dim=hic_hid_dim,
                    latent_dim=latent_dim,
                )
                for i in range(self.n_chr)
            ]
        )

        self.concat_mu = nn.Linear(2 * latent_dim, latent_dim)
        self.concat_logvar = nn.Linear(2 * latent_dim, latent_dim)
        self.mu_encoder = nn.Linear(latent_dim, z_dim)
        # A vMF posterior has a unit direction and one concentration value.
        self.logvar_encoder = nn.Linear(latent_dim, 1)

        if mode != "multi":
            self.rna_only_mu = nn.Linear(latent_dim, latent_dim)
            self.rna_only_logvar = nn.Linear(latent_dim, latent_dim)
            self.hic_only_mu = nn.Linear(latent_dim, latent_dim)
            self.hic_only_logvar = nn.Linear(latent_dim, latent_dim)

    def forward(
        self,
        rna_x=None,
        hic_x=None,
        mode="multi",
        sample_posterior=True,
        return_pre_spherical=False,
    ):
        if mode == "multi":
            rna_mu, rna_logvar = self.RNAencoder(rna_x)
            hic_mu, hic_logvar = self._encode_hic_mean(hic_x)
            concat_mu = self.concat_mu(torch.cat((rna_mu, hic_mu), dim=-1))
            concat_logvar = self.concat_logvar(torch.cat((rna_logvar, hic_logvar), dim=-1))
        elif mode == "rna_only":
            rna_mu, rna_logvar = self.RNAencoder(rna_x)
            concat_mu = self.rna_only_mu(rna_mu)
            concat_logvar = self.rna_only_logvar(rna_logvar)
        elif mode == "hic_only":
            hic_mu, hic_logvar = self._encode_hic_mean(hic_x)
            concat_mu = self.hic_only_mu(hic_mu)
            concat_logvar = self.hic_only_logvar(hic_logvar)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        pre_spherical = self.mu_encoder(concat_mu)
        direction = F.normalize(pre_spherical, dim=-1, eps=1e-8)
        concentration = F.softplus(self.logvar_encoder(concat_logvar)) + 1.0
        concentration = torch.clamp(concentration, min=1.0, max=1e4)
        if sample_posterior:
            z = sample_vmf(direction, concentration)
        else:
            z = direction
        if return_pre_spherical:
            return direction, concentration, z, pre_spherical
        return direction, concentration, z

    def _encode_hic_mean(self, hic_x):
        hic_mus, hic_logvars = [], []
        for i, encoder in enumerate(self.HiCencoders):
            hic_mu_i, hic_logvar_i = encoder(hic_x[i])
            hic_mus.append(hic_mu_i)
            hic_logvars.append(hic_logvar_i)

        hic_mus = torch.stack(hic_mus, dim=1)
        hic_logvars = torch.stack(hic_logvars, dim=1)
        return hic_mus.mean(dim=1), hic_logvars.mean(dim=1)

    def get_hic_shape_info(self):
        conv_sizes = [enc.conv_size for enc in self.HiCencoders]
        hs = [enc.h for enc in self.HiCencoders]
        return conv_sizes, hs


class LibraryEncoder(nn.Module):
    def __init__(
        self,
        library_n_in,
        library_n_out,
        library_n_layers,
        library_n_hidden=128,
        library_dropout_rate=0,
    ):
        super().__init__()
        self.encoder = FCLayers(
            n_in=library_n_in,
            n_out=library_n_hidden,
            n_layers=library_n_layers,
            n_hidden=library_n_hidden,
            dropout_rate=library_dropout_rate,
        )
        self.mu_encoder = nn.Linear(library_n_hidden, library_n_out)
        self.var_encoder = nn.Linear(library_n_hidden, library_n_out)

    def forward(self, x):
        y = self.encoder(x)
        l_mu = self.mu_encoder(y)
        l_logvar = self.var_encoder(y)
        l_logvar = torch.clamp(l_logvar, min=-10, max=10)
        l_logvar = torch.nan_to_num(l_logvar, nan=0.0)
        l_mu = torch.nan_to_num(l_mu, nan=0.0)
        log_l = reparameterize(l_mu, l_logvar)
        l_var = torch.exp(l_logvar) + 1e-6
        return l_mu, l_var, log_l


class CombineDecoder(nn.Module):
    def __init__(
        self,
        rna_output_dim,
        rna_fc_layers,
        rna_hid_dim,
        hic_hid_dim,
        conv_sizes,
        hs,
        z_dim=10,
    ):
        super().__init__()
        self.RNAdecoder = RNADecoder(
            n_output=rna_output_dim,
            n_fc_layers=rna_fc_layers,
            n_hidden=rna_hid_dim,
            z_dim=z_dim,
        )
        self.HiCdecoder = nn.ModuleList(
            [
                HiCDecoder(
                    hid_dim=hic_hid_dim,
                    z_dim=z_dim,
                    conv_size=conv_sizes[i],
                    h=hs[i],
                )
                for i in range(len(conv_sizes))
            ]
        )
        self.n_chr = len(conv_sizes)

    def forward(self, z, log_l, decode_hic=True):
        pred_rna_theta, pred_rna_logit_pi, pred_rna_ratio, pred_rna_rate = self.RNAdecoder(log_l, z)
        pred_hic = []
        if decode_hic:
            z_expand = z.unsqueeze(1).repeat(1, self.n_chr, 1)
            for i, decoder in enumerate(self.HiCdecoder):
                pred_hic.append(decoder(z_expand[:, i, :]))

        return pred_rna_theta, pred_rna_logit_pi, pred_rna_rate, pred_hic


class VAE(nn.Module):
    def __init__(
        self,
        hic_hid_dim,
        rna_input_dim,
        rna_fc_layers,
        rna_hid_dim,
        library_n_in,
        library_n_out,
        library_n_layers,
        library_n_hidden,
        library_dropout_rate,
        rna_output_dim,
        latent_dim,
        z_dim,
        hic_weight,
        n_genes,
        chr_input_dims,
        mode,
        anti_collapse_weight=10.0,
        anti_collapse_target_r=0.6,
    ):
        super().__init__()
        self.z_dim = z_dim
        self.mode = mode

        if chr_input_dims is None:
            chr_input_dims = []

        self.combineEncoder = CombineEncoder(
            rna_input_dim=rna_input_dim,
            rna_fc_layers=rna_fc_layers,
            rna_hid_dim=rna_hid_dim,
            chr_input_dims=chr_input_dims,
            hic_hid_dim=hic_hid_dim,
            latent_dim=latent_dim,
            z_dim=z_dim,
            mode=mode,
        )
        self.libraryEncoder = LibraryEncoder(
            library_n_in,
            library_n_out,
            library_n_layers,
            library_n_hidden,
            library_dropout_rate,
        )
        conv_sizes, hs = self.combineEncoder.get_hic_shape_info()
        self.combineDecoder = CombineDecoder(
            rna_output_dim=rna_output_dim,
            rna_fc_layers=rna_fc_layers,
            rna_hid_dim=rna_hid_dim,
            hic_hid_dim=hic_hid_dim,
            conv_sizes=conv_sizes,
            hs=hs,
            z_dim=z_dim,
        )
        self.hic_weight = hic_weight
        self.n_genes = n_genes
        self.anti_collapse_weight = anti_collapse_weight
        self.anti_collapse_target_r = anti_collapse_target_r

    def anti_collapse_terms(self, mu):
        if mu.shape[0] < 2:
            return mu.new_tensor(0.0), mu.new_tensor(0.0)
    
        mean_direction_norm = torch.linalg.norm(mu.mean(dim=0))
    
        if self.mode != "multi" or self.anti_collapse_weight <= 0:
            return mu.new_tensor(0.0), mean_direction_norm
        hinge_scale = max(1.0 - float(self.anti_collapse_target_r), 1e-6)
        collapse_loss = torch.relu(
            (mean_direction_norm - self.anti_collapse_target_r) / hinge_scale
        ).pow(2)
        return collapse_loss, mean_direction_norm

    def forward(self, rna_x, hic_x, beta, sample_vmf_posterior=True):
        if self.mode == "hic_only":
            if hic_x is None:
                raise ValueError("hic_x cannot be None when mode='hic_only'")

            mu, concentration, z = self.combineEncoder(
                rna_x=None,
                hic_x=hic_x,
                mode="hic_only",
                sample_posterior=sample_vmf_posterior,
            )
            pred_hic = []
            if self.combineDecoder.n_chr > 0:
                z_expand = z.unsqueeze(1).repeat(1, self.combineDecoder.n_chr, 1)
                for i, decoder in enumerate(self.combineDecoder.HiCdecoder):
                    pred_hic.append(decoder(z_expand[:, i, :]))

            if len(pred_hic) > 0:
                raw_hic_loss = hic_loss_fn(hic_x, pred_hic)
                weighted_hic_loss = raw_hic_loss
            else:
                raw_hic_loss = torch.tensor(0.0, device=z.device)
                weighted_hic_loss = torch.tensor(0.0, device=z.device)

            rna_loss = torch.tensor(0.0, device=z.device)
            kl_loss = vmf_kl_uniform(concentration, self.z_dim).mean()
            collapse_loss, collapse_r = self.anti_collapse_terms(mu)
            total_loss = raw_hic_loss + beta * kl_loss + self.anti_collapse_weight * collapse_loss
            return total_loss, rna_loss, raw_hic_loss, weighted_hic_loss, kl_loss, collapse_loss, collapse_r

        if rna_x is None:
            raise ValueError("rna_x cannot be None unless mode='hic_only'")

        rna_x = torch.nan_to_num(rna_x, nan=0.0, posinf=0.0, neginf=0.0)
        rna_sum = torch.clamp(rna_x.sum(1, keepdim=True), min=1e-6)
        log_rna_x = torch.log(rna_sum)

        if self.mode == "rna_only":
            mu, concentration, z = self.combineEncoder(
                rna_x=rna_x,
                hic_x=None,
                mode="rna_only",
                sample_posterior=sample_vmf_posterior,
            )
        else:
            mu, concentration, z = self.combineEncoder(
                rna_x=rna_x,
                hic_x=hic_x,
                mode="multi",
                sample_posterior=sample_vmf_posterior,
            )

        local_l_mu = torch.mean(log_rna_x).unsqueeze(0).float()
        local_l_var = torch.var(log_rna_x).unsqueeze(0).float() + 1e-6

        l_mu, l_var, log_l = self.libraryEncoder(rna_x)
        pred_rna_theta, pred_rna_logit_pi, pred_rna_rate, pred_hic = self.combineDecoder(
            z, log_l, decode_hic=(self.mode != "rna_only")
        )

        rna_weight = 1000 / self.n_genes
        rna_loss = rna_weight * rna_loss_fn(
            rna_x,
            pred_rna_rate,
            pred_rna_theta,
            pred_rna_logit_pi,
            l_mu,
            l_var,
            local_l_mu,
            local_l_var,
        )

        if (self.mode != "rna_only") and (hic_x is not None) and len(pred_hic) > 0:
            raw_hic_loss = hic_loss_fn(hic_x, pred_hic)
            weighted_hic_loss = self.hic_weight * raw_hic_loss
        else:
            raw_hic_loss = torch.tensor(0.0, device=rna_x.device)
            weighted_hic_loss = torch.tensor(0.0, device=rna_x.device)

        kl_loss = vmf_kl_uniform(concentration, self.z_dim).mean()
        collapse_loss, collapse_r = self.anti_collapse_terms(mu)
        total_loss = (
            rna_loss
            + weighted_hic_loss
            + beta * kl_loss
            + self.anti_collapse_weight * collapse_loss
        )
        return total_loss, rna_loss, raw_hic_loss, weighted_hic_loss, kl_loss, collapse_loss, collapse_r

    def get_all_representations(self, rna_x=None, hic_x=None):
        if self.mode == "hic_only":
            mu, concentration, z, pre_spherical = self.combineEncoder(
                rna_x=None,
                hic_x=hic_x,
                mode="hic_only",
                return_pre_spherical=True,
            )
            mu_scaled = mu * torch.log1p(concentration)
            return {
                "mu": mu,
                "pre_spherical": pre_spherical,
                "mu_scaled": mu_scaled,
                "concentration": concentration,
                "z": z,
            }

        if self.mode == "rna_only":
            mu, concentration, z, pre_spherical = self.combineEncoder(
                rna_x=rna_x,
                hic_x=None,
                mode="rna_only",
                return_pre_spherical=True,
            )
            mu_scaled = mu * torch.log1p(concentration)
            return {
                "mu": mu,
                "pre_spherical": pre_spherical,
                "mu_scaled": mu_scaled,
                "concentration": concentration,
                "z": z,
            }

        mu, concentration, z, pre_spherical = self.combineEncoder(
            rna_x=rna_x,
            hic_x=hic_x,
            mode="multi",
            return_pre_spherical=True,
        )
        mu_scaled = mu * torch.log1p(concentration)
        rna_mu, _ = self.combineEncoder.RNAencoder(rna_x)
        hic_mu, _ = self.combineEncoder._encode_hic_mean(hic_x)
        return {
            "rna_mu": rna_mu,
            "hic_mu": hic_mu,
            "mu": mu,
            "pre_spherical": pre_spherical,
            "mu_scaled": mu_scaled,
            "concentration": concentration,
            "z": z,
        }
