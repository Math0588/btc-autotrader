# 🤖 Polymarket BTC Autotrader — Guide de Déploiement

## Architecture

```
autotrader.py          ← Bot autonome (scan toutes les 15 min)
autotrader_state.json  ← État persisté (positions, PnL, etc.)
autotrader.log         ← Logs du bot
render.yaml            ← Config déploiement Render.com
Dockerfile             ← Container Docker
```

## 🚀 Déploiement Gratuit sur Render.com

### Étape 1 : Créer un Repo GitHub

```bash
cd btc_options_surface
git init
git add autotrader.py requirements.txt render.yaml Dockerfile
git commit -m "Autotrader deployment"
# Créer un repo sur github.com, puis :
git remote add origin https://github.com/TON_USER/btc-autotrader.git
git push -u origin main
```

### Étape 2 : Déployer sur Render

1. Créer un compte sur [render.com](https://render.com) (gratuit)
2. Cliquer **"New +"** → **"Background Worker"**
3. Connecter ton repo GitHub
4. Runtime: **Python 3**
5. Build Command: `pip install -r requirements.txt`
6. Start Command: `python -u autotrader.py`
7. Plan: **Free**
8. Ajouter les variables d'environnement :
   - `STARTING_CAPITAL` = `100`
   - `SCAN_INTERVAL` = `900` (15 min)
   - `MIN_EDGE` = `3.0`
   - `TELEGRAM_TOKEN` = *(optionnel, voir ci-dessous)*
   - `TELEGRAM_CHAT` = *(optionnel)*

### ⚠️ Limitation Render Free Tier
Le free tier de Render suspend les workers après ~15 min d'inactivité.
**Solution** : Utiliser [UptimeRobot](https://uptimerobot.com) pour pinger le service.

---

## 🔔 Notifications Telegram (Optionnel mais recommandé)

### Créer un Bot Telegram
1. Ouvrir Telegram, chercher **@BotFather**
2. Envoyer `/newbot`
3. Donner un nom (ex: `BTC Autotrader`)
4. Copier le **token** (format: `123456:ABC-DEF...`)
5. Ouvrir une conversation avec ton bot
6. Aller sur `https://api.telegram.org/bot<TOKEN>/getUpdates`
7. Envoyer un message au bot, puis refresh la page
8. Copier le `chat_id` du résultat

### Configurer les Variables
```
TELEGRAM_TOKEN=123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11
TELEGRAM_CHAT=987654321
```

---

## 📊 Stratégie en Détail

### Ce que fait le bot :
1. **Toutes les 15 min** → Scan 66+ marchés Polymarket BTC
2. **Récupère l'IV Deribit** → 900+ data points en 1 seul appel API
3. **Calcule les probabilités** → Modèle European + One-Touch blended
4. **Identifie l'edge** → Compare modèle vs prix Polymarket
5. **Exécute les trades** → Max 3 par scan, Kelly sizing conservatif
6. **Gère le portfolio** → Settlement automatique à expiry, drawdown control

### Paramètres de risque :
- **Kelly Fraction** : 20% (très conservatif)
- **Max par trade** : 15% du capital
- **Max exposition** : 80% du capital
- **Edge minimum** : 3%
- **Win prob minimum** : 15%
- **Drawdown control** : Réduit le sizing à 50% au-delà de 30% de drawdown

### Exemple avec €100 :
```
Scan #1: Identifie 10 opportunités
  → Trade 1: BTC SOUS $60k (5j) → $1.49 @ 13.5¢/contrat  Win: 21%  Gain si win: +628%
  → Trade 2: BTC AU-DESSUS $66k (5j) → $0.90 @ 30¢/contrat  Win: 34%  Gain si win: +229%
  → Trade 3: BTC SOUS $58k (7j) → $1.35 @ 9¢/contrat  Win: 16%  Gain si win: +991%
  
Capital restant: $96.26 | Exposition: $3.74 | 3 positions ouvertes
```

---

## 🔧 Run Local

```bash
# Installation
pip install -r requirements.txt

# Test (un seul scan)
python autotrader.py --once

# Run continu (15 min entre scans)
python autotrader.py

# Avec Telegram
TELEGRAM_TOKEN=xxx TELEGRAM_CHAT=yyy python autotrader.py

# Capital personnalisé
STARTING_CAPITAL=200 python autotrader.py
```

## ⚠️ Avertissement

Ce bot est en **mode PAPER TRADING** par défaut (simulation).
Il ne place PAS de vrais ordres sur Polymarket.
Pour du trading réel, il faudrait intégrer le `py-clob-client` de Polymarket
avec vos clés API et un wallet USDC sur Polygon.

**Le trading comporte des risques. Ceci n'est pas un conseil financier.**
