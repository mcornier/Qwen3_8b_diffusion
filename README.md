# Qwen3_8b_diffusion
Try to make a token prediction by diffusion version of Qwen3_8b

---

# Spécification Technique : Prototype LLM × Diffusion

*Version 1.0 - Implémentation complète*

---

## 1. Vue d'ensemble du système

### 1.1 Objectif
Implémenter un système hybride combinant un LLM gelé (Qwen3-8B) avec un mécanisme de diffusion pour générer du texte token par token via des "patchs progressifs".

### 1.2 Architecture globale

```
INPUT: Prompt text
    ↓
[Tokenizer] → tokens
    ↓
[Qwen3-8B Frozen] → h_t (4096d context embedding)
    ↓
[Patch Generation Loop]
    ├── Active Patches: K patches simultanés (chacun avec t_i individuel)
    ├── DiffusionHead: ε prediction sur chaque patch
    ├── Scheduler: x_t → x_{t-1} denoising (t_i décrémente)
    ├── Latent→Token Decoder: patch → nouveau token
    └── Token → Qwen KV Cache Update → nouveau h_t (CYCLE FEEDBACK)
    ↓
OUTPUT: Generated text (token stream)
```

### 1.3 Schéma « stagger » (1 nouveau patch par step)

| **Global step** | Patch 0     | Patch 1     | Patch 2     | … | Action                                |
| --------------- | ----------- | ----------- | ----------- | - | ------------------------------------- |
|  0              | t=10        | –           | –           |   | création patch 0                      |
|  1              | t=9         | t=10        | –           |   | + patch 1                             |
|  2              | t=8         | t=9         | t=10        |   | + patch 2                             |
|  …              | …           | …           | …           |   | diffusion parallèle                   |
|  10             | t=0 → token | t=1         | t=2         |   | patch 0 terminé, décodé, token → Qwen |
|  11             | –           | t=0 → token | t=1         |   | patch 1 terminé                       |
|  12             | –           | –           | t=0 → token |   | patch 2 terminé (fenêtre vide)        |

> **Chaque patch effectue exactement `T_init` passes**, mais son calendrier est décalé.

---

## 2. Composants détaillés

### 2.1 Qwen3-8B (Backbone gelé)

**Spécifications :**
- Modèle : Qwen3-8B quantifié (int8 ou fp8)
- État : Gelé (pas d'entraînement)
- Sortie : Embeddings contextuels h_t de dimension 4096
- Cache KV : Maintenu pour génération incrémentale O(1)

**Implémentation :**
```python
class FrozenQwen3:
    def __init__(self, model_path: str, quantization: str = "int8"):
        self.model = load_quantized_qwen3(model_path, quantization)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        
        self.past_key_values = None
        self.context_length = 0
    
    def forward_incremental(self, new_tokens: torch.Tensor) -> torch.Tensor:
        """Forward pass incrémental avec cache KV"""
        with torch.no_grad():
            outputs = self.model(
                new_tokens, 
                past_key_values=self.past_key_values,
                use_cache=True
            )
            
            # Récupérer les nouveaux past_key_values pour le prochain appel
            self.past_key_values = outputs.past_key_values
            self.context_length += new_tokens.size(1)
            
            # Retourner les hidden states
            return outputs.last_hidden_state  # [batch_size, seq_len, 4096]
```

### 2.2 Image Autoencoder CNN (pour créer des latents "image-like")

**Objectif :** Créer des latents de référence "Image Réelle" pour entraîner le NoiseClassifier

**Dimension latente :** 5120D (pour capturer les détails des tokens Qwen3-8B 4096D)

**Options d'architecture à tester :**

#### Option A : 640×8 (3 couches CNN)
```python
class ImageAutoencoder_640x8(nn.Module):
    """
    Architecture 3 couches CNN : 640×8×3 → 5120D
    Moins de couches = reconstruction plus facile
    """
    def __init__(self, latent_dim: int = 5120):
        super().__init__()
        
        # Encodeur CNN: [3, 640, 8] → [5120]
        self.encoder = nn.Sequential(
            # Conv1: 3→16 canaux, stride=2 → [16, 320, 4]
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            # Conv2: 16→40 canaux, stride=2 → [40, 160, 2]  
            nn.Conv2d(16, 40, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            # Conv3: 40→16 canaux, stride=2 → [16, 80, 1]
            nn.Conv2d(40, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            # Flatten: 16 * 80 * 1 = 1280 → Linear → 5120
            nn.Flatten(),
            nn.Linear(1280, latent_dim)
        )
        
        # Décodeur CNN transposé: [5120] → [3, 640, 8]
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 1280),
            nn.Unflatten(1, (16, 80, 1)),
            # Deconv1: 16→40 canaux
            nn.ConvTranspose2d(16, 40, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            # Deconv2: 40→16 canaux  
            nn.ConvTranspose2d(40, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            # Deconv3: 16→3 canaux
            nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Tanh()
        )
```

#### Option B : 320×16 (4 couches CNN)
```python
class ImageAutoencoder_320x16(nn.Module):
    """
    Architecture 4 couches CNN : 320×16×3 → 5120D
    Plus de couches = plus d'information spatiale mais reconstruction plus difficile
    """
    def __init__(self, latent_dim: int = 5120):
        super().__init__()
        
        # Encodeur CNN: [3, 320, 16] → [5120]
        self.encoder = nn.Sequential(
            # Conv1: 3→8 canaux, stride=2 → [8, 160, 8]
            nn.Conv2d(3, 8, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            # Conv2: 8→16 canaux, stride=2 → [16, 80, 4]
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            # Conv3: 16→32 canaux, stride=2 → [32, 40, 2]
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            # Conv4: 32→64 canaux, stride=2 → [64, 20, 1]
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            # Flatten: 64 * 20 * 1 = 1280 → Linear → 5120
            nn.Flatten(),
            nn.Linear(1280, latent_dim)
        )
        
        # Décodeur correspondant (4 couches transposées)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 1280),
            nn.Unflatten(1, (64, 20, 1)),
            # 4 couches deconv symétriques
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(8, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Tanh()
        )
```

#### Option C : 1280×4×1 (2 couches CNN - dernier recours)
```python
class ImageAutoencoder_1280x4x1(nn.Module):
    """
    Architecture 2 couches CNN : 1280×4×1×3 → 5120D
    Minimum de couches CNN = reconstruction la plus facile
    """
    def __init__(self, latent_dim: int = 5120):
        super().__init__()
        
        # Encodeur CNN: [3, 1280, 4] → [5120]
        self.encoder = nn.Sequential(
            # Conv1: 3→20 canaux, stride=2 → [20, 640, 2]
            nn.Conv2d(3, 20, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            # Conv2: 20→128 canaux, stride=2 → [128, 320, 1]
            nn.Conv2d(20, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            # Flatten: 128 * 320 * 1 = 40960 → Linear → 5120
            nn.Flatten(),
            nn.Linear(40960, latent_dim)
        )
        
        # Décodeur CNN transposé: [5120] → [3, 1280, 4]
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 40960),
            nn.Unflatten(1, (128, 320, 1)),
            # Deconv1: 128→20 canaux
            nn.ConvTranspose2d(128, 20, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            # Deconv2: 20→3 canaux
            nn.ConvTranspose2d(20, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Tanh()
        )
```

### 2.3 Token-to-Latent Autoencoder

**Architecture :**
- Encoder : Token embeddings (32k vocab) → Latent 5120D normalisé [-1,1]
- Decoder : Latent 5120D → Token logits (32k vocab)
- Dimension intermédiaire : 512 → 1024 → 5120
- **Normalisation** : Latents compatibles avec le noise scheduler

**Implémentation :**
```python
class TokenLatentAutoencoder(nn.Module):
    def __init__(self, vocab_size=32000, latent_dim=4096):
        super().__init__()
        
        # Encoder: Token → Latent
        self.token_embedding = nn.Embedding(vocab_size, 512)
        self.encoder = nn.Sequential(
            nn.Linear(512, 768),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(768, latent_dim),
            nn.LayerNorm(latent_dim)
        )
        
        # Normalisation pour compatibilité diffusion (échelle [0,1])
        self.latent_normalizer = nn.Tanh()
        
        # Decoder: Latent → Token logits
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 768),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Linear(512, vocab_size)
        )
    
    def encode(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens [batch_size, seq_len] → latents [batch_size, seq_len, 4096] normalisés [0,1]"""
        embedded = self.token_embedding(tokens)
        latents = self.encoder(embedded)
        # Normaliser pour la diffusion
        return self.latent_normalizer(latents)
    
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """latents [batch_size, seq_len, 4096] → logits [batch_size, seq_len, vocab_size]"""
        return self.decoder(latents)
    
    def encode_with_noise(self, tokens: torch.Tensor, noise_scheduler, 
                         timesteps: torch.Tensor) -> tuple:
        """Encoder avec ajout de bruit pour l'entraînement"""
        # Encoder sans bruit
        clean_latents = self.encode(tokens)
        
        # Générer du bruit gaussien
        noise = torch.randn_like(clean_latents)
        
        # Ajouter du bruit selon le schedule
        noisy_latents = noise_scheduler.add_noise(clean_latents, noise, timesteps)
        
        return noisy_latents, noise, clean_latents
```

### 2.3 Noise Classifier (adapté du projet existant)

**Objectif :** Classifier le contenu latent en 4 classes pour déterminer si un latent est "décodable"

**Classes :**
- 0: Bruit Image
- 1: Bruit Latent  
- 2: Couleur Unie
- 3: Image Réelle (= "décodable")

**Architecture :** MLP → Attention → MLP (identique au projet actuel)

```python
class LatentContentClassifier(nn.Module):
    """
    Classifier de contenu latent adapté pour les tokens 4096d.
    Basé sur l'architecture existante du projet mais adapté aux dimensions 4096d.
    """
    
    def __init__(
        self,
        input_dim: int = 4096,  # Adapté pour les latents 4096d
        hidden_dim: int = 512,
        num_classes: int = 4,   # [Bruit Image, Bruit Latent, Couleur Unie, Image Réelle]
        num_heads: int = 8,
        dropout: float = 0.1,
        use_attention: bool = True
    ):
        super().__init__()
        
        # MLP1 : Feature Extraction (4096 → 512 → 512)
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # Couche d'attention (optionnelle)
        if use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            )
        
        # MLP2 : Classification Head (512 → 256 → 4)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
        
        self.use_attention = use_attention
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: Latents [batch_size, seq_len, 4096] ou [batch_size, 4096]
        Returns:
            Dict avec 'logits', 'probabilities', 'is_decodable'
        """
        original_shape = x.shape
        if len(original_shape) == 3:
            # Reshape pour traitement par batch
            batch_size, seq_len, dim = original_shape
            x = x.view(-1, dim)  # [batch_size * seq_len, 4096]
        
        # Feature extraction
        features = self.feature_extractor(x)  # [N, 512]
        
        # Attention (si activée)
        if self.use_attention:
            features_seq = features.unsqueeze(1)  # [N, 1, 512]
            attended_features, _ = self.attention(features_seq, features_seq, features_seq)
            features = attended_features.squeeze(1)  # [N, 512]
        
        # Classification
        logits = self.classifier(features)  # [N, 4]
        probabilities = torch.softmax(logits, dim=-1)
        
        # Déterminer si "décodable" (classe 3 = Image Réelle)
        is_decodable = probabilities[:, 3] > 0.5  # Seuil à 50%
        
        # Restaurer la forme originale si nécessaire
        if len(original_shape) == 3:
            logits = logits.view(batch_size, seq_len, 4)
            probabilities = probabilities.view(batch_size, seq_len, 4)
            is_decodable = is_decodable.view(batch_size, seq_len)
        
        return {
            'logits': logits,
            'probabilities': probabilities,
            'is_decodable': is_decodable
        }
    
    def is_decodable(self, latents: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """
        Méthode utilitaire pour déterminer si les latents sont décodables.
        
        Args:
            latents: [batch_size, seq_len, 4096] ou [batch_size, 4096]
            threshold: Seuil de probabilité pour la classe "Image Réelle"
        
        Returns:
            torch.Tensor: Masque booléen indiquant les latents décodables
        """
        with torch.no_grad():
            outputs = self.forward(latents)
            # Classe 3 = "Image Réelle" = décodable
            return outputs['probabilities'][..., 3] > threshold
```

**Utilisation dans l'entraînement :**
- **Phase 1** : Entraîner le classifier sur des données synthétiques (bruit vs latents propres)
- **Phase 2** : Utiliser le classifier pour valider que les latents du Token-AE sont classifiés comme "Image Réelle" (classe 3)
- **Phase 3** : Utiliser pour l'early-exit optionnel dans la diffusion

### 2.4 TimestepEmbedding

**Implémentation de l'embedding temporel sinusoïdal :**

```python
class TimestepEmbedding(nn.Module):
    """Embedding des timesteps de diffusion avec encodage sinusoïdal."""
    
    def __init__(self, embedding_dim: int, max_timesteps: int = 1000):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.max_timesteps = max_timesteps
        
        # Projection pour transformer l'encodage sinusoïdal
        self.projection = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 4),
            nn.SiLU(),
            nn.Linear(embedding_dim * 4, embedding_dim)
        )
    
    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            timesteps: [batch_size] ou [batch_size, K]
            
        Returns:
            embeddings: [batch_size, embedding_dim] ou [batch_size, K, embedding_dim]
        """
        original_shape = timesteps.shape
        timesteps = timesteps.flatten().unsqueeze(-1).float()
        
        # Encodage sinusoïdal
        half_dim = self.embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        emb = timesteps * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        
        # Projection
        emb = self.projection(emb)
        
        # Restaurer la forme originale
        if len(original_shape) == 1:
            return emb  # [batch_size, embedding_dim]
        else:
            return emb.view(*original_shape, self.embedding_dim)  # [batch_size, K, embedding_dim]
```

### 2.5 DiffusionHead

**Architecture :** Transformer spécialisé pour prédiction de bruit

```python
class DiffusionHead(nn.Module):
    def __init__(self, d_model=4096, n_layers=16, n_heads=32, d_ff=16384):
        super().__init__()
        
        # Time embedding
        self.time_embedding = TimestepEmbedding(d_model)
        
        # Transformer layers avec cross-attention explicite
        self.layers = nn.ModuleList([
            CrossAttentionTransformerLayer(d_model, n_heads, d_ff, dropout=0.1)
            for _ in range(n_layers)
        ])
        
        # Output projection pour ε
        self.epsilon_proj = nn.Linear(d_model, d_model)
        
    def forward(self, x_t: torch.Tensor, t: torch.Tensor, 
                context: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_t: Patches bruités [batch_size, K, 4096]
            t: Timesteps [batch_size, K] - timesteps individuels par patch
            context: Context du LLM [batch_size, seq_len, 4096]
        Returns:
            epsilon: Bruit prédit [batch_size, K, 4096]
        """
        # Time embedding pour chaque patch individuellement
        time_emb = self.time_embedding(t)  # [batch_size, K, 4096]
        
        # Ajouter time embedding aux patches
        x = x_t + time_emb
        
        # Traitement par les couches transformer avec cross-attention
        for layer in self.layers:
            x = layer(x, context=context)  # Q=patches, K,V=context
        
        # Prédiction du bruit
        epsilon = self.epsilon_proj(x)
        return epsilon


class CrossAttentionTransformerLayer(nn.Module):
    """Couche Transformer avec self-attention et cross-attention vers le contexte."""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        
        # Self-attention sur les patches
        self.self_attention = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        
        # Cross-attention vers le contexte
        self.cross_attention = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        
        # Feed-forward
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        # Normalisation
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, patches: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Args:
            patches: [batch_size, K, d_model] patches de diffusion
            context: [batch_size, seq_len, d_model] contexte LLM
        """
        # Self-attention sur les patches
        attn_output, _ = self.self_attention(patches, patches, patches)
        patches = self.norm1(patches + self.dropout(attn_output))
        
        # Cross-attention vers le contexte (Q=patches, K,V=context)
        cross_attn_output, _ = self.cross_attention(patches, context, context)
        patches = self.norm2(patches + self.dropout(cross_attn_output))
        
        # Feed-forward
        ff_output = self.feed_forward(patches)
        patches = self.norm3(patches + self.dropout(ff_output))
        
        return patches
```

### 2.5 Noise Scheduler

**Implémentation du scheduler cosine :**

```python
class CosineNoiseScheduler:
    def __init__(self, num_timesteps=1000, s=0.008):
        self.num_timesteps = num_timesteps
        self.s = s
        
        # Cosine schedule
        steps = torch.arange(num_timesteps + 1, dtype=torch.float32)
        alphas_cumprod = torch.cos(((steps / num_timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        self.betas = torch.clip(betas, 0, 0.999)
        
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
        # Précalculs
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
    
    def add_noise(self, x_0: torch.Tensor, noise: torch.Tensor, 
                  timesteps: torch.Tensor) -> torch.Tensor:
        """Ajouter du bruit selon le schedule"""
        sqrt_alpha_prod = self.sqrt_alphas_cumprod[timesteps]
        sqrt_one_minus_alpha_prod = self.sqrt_one_minus_alphas_cumprod[timesteps]
        
        return (sqrt_alpha_prod.unsqueeze(-1) * x_0 + 
                sqrt_one_minus_alpha_prod.unsqueeze(-1) * noise)
    
    def step(self, model_output: torch.Tensor, timestep: int, 
             sample: torch.Tensor) -> torch.Tensor:
        """Un pas de débruitage"""
        # Implémentation DDIM simplifiée
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[timestep - 1] if timestep > 0 else 1.0
        
        beta_prod_t = 1 - alpha_prod_t
        
        # Prédire x_0
        pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        
        # Direction vers x_t
        pred_sample_direction = (1 - alpha_prod_t_prev) ** 0.5 * model_output
        
        # x_{t-1}
        prev_sample = alpha_prod_t_prev ** 0.5 * pred_original_sample + pred_sample_direction
        
        return prev_sample
```

---

## 3. Pipeline de génération temps-réel

### 3.1 Cycle d'inférence principal

```python
class DiffusionTextGenerator:
    def __init__(self, qwen_model, token_ae, diffusion_head, scheduler, 
                 noise_classifier, K_max=8, T_init=10):
        self.qwen = qwen_model
        self.token_ae = token_ae
        self.diffusion_head = diffusion_head
        self.scheduler = scheduler
        self.noise_classifier = noise_classifier
        
        self.K_max = K_max  # Nombre max de patches simultanés
        self.T_init = T_init  # Timesteps initiaux
        
    def generate(self, prompt: str, max_length: int = 100) -> str:
        """Génération de texte avec patchs progressifs"""
        
        # 1. Tokenisation et contexte initial
        tokens = self.tokenizer.encode(prompt)
        context_tokens = torch.tensor(tokens).unsqueeze(0)
        
        # 2. Forward pass Qwen pour contexte initial
        h_t = self.qwen.forward_incremental(context_tokens)
        
        # 3. Initialisation des patches
        active_patches = []
        generated_tokens = []
        
        # Créer K_0 patches initiaux
        K_0 = min(self.K_max, max_length)
        for i in range(K_0):
            patch = {
                'x_t': torch.randn(1, 1, 4096),  # Bruit initial
                't_i': self.T_init,  # Timestep initial
                'id': i
            }
            active_patches.append(patch)
        
        # 4. Boucle de génération
        generating = True
        while active_patches and generating:
            
            # 4a. Préparer les données pour le batch
            batch_x_t = torch.cat([p['x_t'] for p in active_patches], dim=1)
            batch_t = torch.tensor([p['t_i'] for p in active_patches])
            
            # 4b. Prédiction du bruit par DiffusionHead
            epsilon_pred = self.diffusion_head(
                x_t=batch_x_t,
                t=batch_t,
                context=h_t
            )
            
            # 4c. Débruitage avec scheduler
            patches_to_remove = []
            for i, patch in enumerate(active_patches):
                # Un pas de débruitage
                patch['x_t'] = self.scheduler.step(
                    model_output=epsilon_pred[:, i:i+1, :],
                    timestep=patch['t_i'],
                    sample=patch['x_t']
                )
                patch['t_i'] -= 1
                
                # 4d. Si patch terminé (t_i == 0)
                if patch['t_i'] == 0:
                    # Décoder le patch en token
                    logits = self.token_ae.decode(patch['x_t'])
                    token = torch.argmax(logits, dim=-1).item()
                    
                    # 4e. Vérifier EOS - FLUSH CRITIQUE
                    if token == self.tokenizer.eos_token_id:
                        # PURGE IMMÉDIATE : vider tous les patches actifs
                        active_patches.clear()
                        generating = False
                        print(f"[EOS] Token EOS détecté, purge de tous les patches actifs")
                        break
                    
                    # 4f. Ajouter le token généré
                    generated_tokens.append(token)
                    patches_to_remove.append(i)
                    
                    # 4g. Mise à jour incrémentale du contexte Qwen
                    new_token = torch.tensor([[token]])
                    h_t = self.qwen.forward_incremental(new_token)
            
            # Supprimer les patches terminés (sauf si EOS a tout purgé)
            if generating:  # Seulement si pas d'EOS
                for i in reversed(patches_to_remove):
                    active_patches.pop(i)
            
            # 4h. Créer de nouveaux patches si nécessaire
            if generating and len(active_patches) < self.K_max:
                num_new = min(self.K_max - len(active_patches), 
                             max_length - len(generated_tokens))
                
                for _ in range(num_new):
                    new_patch = {
                        'x_t': torch.randn(1, 1, 4096),
                        't_i': self.T_init,
                        'id': len(generated_tokens) + len(active_patches)
                    }
                    active_patches.append(new_patch)
        
        # 5. Décoder les tokens générés
        return self.tokenizer.decode(generated_tokens)
```

---

## 4. Entraînement (Teacher-Student)

### 4.1 Génération des données d'entraînement

```python
class TeacherStudentDataGenerator:
    def __init__(self, teacher_model, token_ae, max_depth=2):
        self.teacher = teacher_model  # Qwen3-8B non quantifié
        self.token_ae = token_ae
        self.max_depth = max_depth
    
    def generate_training_data(self, prompts: List[str]) -> Dict:
        """Générer des données d'entraînement avec branches teacher"""
        
        training_samples = []
        
        for prompt in prompts:
            # 1. Forward pass teacher
            tokens = self.tokenizer.encode(prompt)
            with torch.no_grad():
                teacher_output = self.teacher(torch.tensor(tokens).unsqueeze(0))
                teacher_logits = teacher_output.logits
            
            # 2. Générer branches avec top-k sampling
            branches = self._generate_branches(teacher_logits, k=2, depth=self.max_depth)
            
            # 3. Pour chaque branche, créer des échantillons d'entraînement
            for branch_tokens, branch_prob in branches:
                # Encoder tokens en latents
                token_latents = self.token_ae.encode(torch.tensor(branch_tokens))
                
                # Générer timesteps aléatoires
                timesteps = torch.randint(0, 1000, (len(branch_tokens),))
                
                # Ajouter du bruit
                noise = torch.randn_like(token_latents)
                noisy_latents = self.scheduler.add_noise(token_latents, noise, timesteps)
                
                # Créer échantillon d'entraînement
                sample = {
                    'context_tokens': torch.tensor(tokens),
                    'target_tokens': torch.tensor(branch_tokens),
                    'noisy_latents': noisy_latents,
                    'target_noise': noise,
                    'timesteps': timesteps,
                    'teacher_logits': teacher_logits,
                    'branch_probability': branch_prob
                }
                training_samples.append(sample)
        
        return training_samples
    
    def _generate_branches(self, logits: torch.Tensor, k: int, depth: int) -> List[Tuple]:
        """Générer des branches de tokens avec probabilités"""
        branches = []
        
        def _recursive_branch(current_tokens, current_prob, remaining_depth):
            if remaining_depth == 0:
                branches.append((current_tokens, current_prob))
                return
            
            # Top-k sur le dernier token
            last_logits = logits[0, len(current_tokens)-1, :]
            top_k_probs, top_k_indices = torch.topk(torch.softmax(last_logits, dim=-1), k)
            
            for prob, token_id in zip(top_k_probs, top_k_indices):
                new_tokens = current_tokens + [token_id.item()]
                new_prob = current_prob * prob.item()
                _recursive_branch(new_tokens, new_prob, remaining_depth - 1)
        
        # Commencer la récursion
        first_token_logits = logits[0, -1, :]
        top_k_probs, top_k_indices = torch.topk(torch.softmax(first_token_logits, dim=-1), k)
        
        for prob, token_id in zip(top_k_probs, top_k_indices):
            _recursive_branch([token_id.item()], prob.item(), depth - 1)
        
        return branches
```

### 4.2 Loss function composite

```python
class CompositeDiffusionLoss(nn.Module):
    def __init__(self, lambda_kl=0.1):
        super().__init__()
        self.lambda_kl = lambda_kl
        self.mse_loss = nn.MSELoss()
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
    
    def forward(self, predicted_epsilon, target_noise, predicted_x0, 
                teacher_logits, importance_weights):
        """
        Loss composite : MSE(ε) + λ·KL(decoder(x0_pred) || teacher_logits)
        """
        
        # 1. Loss de diffusion (MSE sur le bruit)
        diffusion_loss = self.mse_loss(predicted_epsilon, target_noise)
        
        # 2. Loss de fidélité au teacher (KL divergence)
        student_logits = predicted_x0  # Approximation
        teacher_probs = torch.softmax(teacher_logits, dim=-1)
        student_log_probs = torch.log_softmax(student_logits, dim=-1)
        
        kl_loss = self.kl_loss(student_log_probs, teacher_probs)
        
        # 3. Pondération par importance sampling
        weighted_diffusion_loss = (diffusion_loss * importance_weights).mean()
        
        # 4. Loss totale
        total_loss = weighted_diffusion_loss + self.lambda_kl * kl_loss
        
        return {
            'total_loss': total_loss,
            'diffusion_loss': weighted_diffusion_loss,
            'kl_loss': kl_loss
        }
```

---

## 5. Configuration et hyperparamètres

### 5.1 Hyperparamètres par défaut

```yaml
model:
  qwen:
    model_path: "Qwen/Qwen3-8B"
    quantization: "int8"  # ou "fp8"
    
  token_ae:
    vocab_size: 32000
    latent_dim: 4096
    
  diffusion_head:
    d_model: 4096
    n_layers: 16
    n_heads: 32
    d_ff: 16384
    dropout: 0.1
    
  noise_classifier:
    latent_dim: 4096
    
generation:
  K_max: 8          # Patches simultanés max
  K_0: 8            # Patches initiaux
  T_init: 10        # Timesteps initiaux
  max_length: 512   # Longueur max de génération
  
training:
  batch_size: 4     # Petit batch pour 4096d
  learning_rate: 1e-4
  weight_decay: 0.01
  epochs: 100
  warmup_steps: 1000
  gradient_clip: 1.0
  
  # Loss composite
  lambda_kl: 0.1
  
  # Teacher-student
  max_branch_depth: 2
  top_k: 2
  
scheduler:
  num_timesteps: 1000
  schedule_type: "cosine"
  s: 0.008
```

### 5.2 Variantes à tester

```yaml
experiments:
  T_init_variants: [6, 8, 10, 12]
  n_layers_variants: [12, 16, 20]
  K_variants: [4, 8, 16]
  sampler_variants: ["DDIM", "DPM-Solver"]
  batch_tokens_kv: [1, 2, 4]
```

---

## 6. Séquence d'entraînement (ordre critique)

### 6.1 Ordre des prérequis

```
1. NoiseClassifier train (labels: {noise≤τ = decodable})
   ↓
2. Token-AE train (reconstruction + noise compatibility)  
   ↓
3. DiffusionHead + composite loss
```

**Justification :** Le NoiseClassifier doit être entraîné AVANT le Token-AE pour fournir les labels "décodable" nécessaires à l'entraînement de l'autoencoder.

### 6.2 Roadmap d'implémentation

### Sprint 0 (1 semaine) : Skeleton + KV Cache
- [ ] Implémentation FrozenQwen3 avec cache KV correct
- [ ] Tests de performance incrémentale O(1)
- [ ] Structure de base du pipeline
- [ ] Validation cache KV avec `past_key_values`

### Sprint 1 (1 semaine) : NoiseClassifier → Token-AE
- [ ] **Phase 1a** : Implémentation et entraînement NoiseClassifier
  - [ ] Génération de données synthétiques (bruit gaussien vs latents propres)
  - [ ] Entraînement classification binaire avec seuil τ
  - [ ] Validation : précision ≥ 95% sur données test
- [ ] **Phase 1b** : Implémentation TokenLatentAutoencoder
  - [ ] Entraînement reconstruction token→latent→token
  - [ ] Validation avec NoiseClassifier : latents "diffusables"
  - [ ] Tests de qualité : reconstruction ≥ 90%

### Sprint 2 (2 semaines) : DiffusionHead convergence
- [ ] Implémentation DiffusionHead avec TimestepEmbedding explicite
- [ ] Système d'entraînement teacher-student avec cut-off probabilité
- [ ] Loss composite avec KL divergence (λ=0.1)
- [ ] Validation convergence sur données synthétiques
- [ ] Tests avec mini-prototype (hidden 1024 → 4096)

### Sprint 3 (2 semaines) : Pipeline complet
- [ ] Intégration complète du pipeline de génération
- [ ] Implémentation schéma "stagger" avec timesteps individuels
- [ ] Gestion EOS avec purge immédiate des patches
- [ ] Tests avec K=8, qualité ≥ baseline Qwen3-8B
- [ ] Optimisations performance (Flash Attention, mixed precision)
- [ ] Métriques d'évaluation (perplexité, BLEU, vitesse tokens/sec)

---

## 7. Questions ouvertes et décisions techniques

### 7.1 Choix de quantization
- **int8 AWQ** : Plus compatible, moins de mémoire
- **fp8 H100** : Plus rapide sur hardware récent
- **Recommandation** : Commencer avec int8, migrer vers fp8 si H100 disponible

### 7.2 Fenêtre K_max
- **K=8** : 8 tokens en parallèle (baseline)
- **K=64** : 64 tokens en parallèle (objectif ambitieux)
- **Recommandation** : Commencer K=8, augmenter progressivement

### 7.3 Métriques d'évaluation
- **Perplexité** : Qualité du modèle de langage
- **BLEU/ROUGE** : Qualité de génération
- **CLIP-Score** : Cohérence texte-latent
- **Vitesse** : Tokens/seconde vs baseline

Cette spécification fournit une roadmap complète pour implémenter le prototype LLM × Diffusion.
