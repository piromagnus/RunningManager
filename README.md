# RunningManager

A Streamlit multi-page application for managing trail running coaching for coaches and athletes.

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) package manager (recommended)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd RunningManager
   ```

2. **Install dependencies**
   ```bash
   uv sync
   ```

3. **Set up environment variables** (see [Environment Setup](#environment-setup) below)

4. **Run the application**
   ```bash
   uv run streamlit run app.py
   ```

## 🔧 Environment Setup

The application requires several environment variables to function properly. These are stored in a `.env` file in the project root.

### Creating the .env file

1. **Copy the example file**
   ```bash
   cp .env.example .env
   ```

2. **Edit the `.env` file** with your actual values (see sections below)

### Required Environment Variables

#### Data Directory (Optional)

```bash
DATA_DIR=./data
```

- **Purpose**: Specifies where the application stores CSV data files
- **Default**: `./data` (relative to project root)
- **Optional**: Can be omitted if you want to use the default

#### Strava API Configuration (Required for Strava Integration)

```bash
STRAVA_CLIENT_ID=your_actual_client_id
STRAVA_CLIENT_SECRET=your_actual_client_secret
STRAVA_REDIRECT_URI=http://localhost:8501/callback
```

**Getting Strava API Credentials:**

1. Go to [Strava API Settings](https://www.strava.com/settings/api)
2. Click "Create App" or use an existing app
3. Fill in the required details:
   - **Application Name**: RunningManager (or your preferred name)
   - **Category**: Other
   - **Club/Team Website**: Your website (can be localhost for development)
   - **Application Website**: Your website (can be localhost for development)
   - **Callback Domain(s)**: `localhost` (for development)

4. **Copy the credentials**:
   - `Client ID` → `STRAVA_CLIENT_ID`
   - `Client Secret` → `STRAVA_CLIENT_SECRET`

5. **Set the redirect URI**:
   - For development: `http://localhost:8501/callback`
   - For production: Use your deployed application's callback URL
   - Must match exactly what's registered in your Strava app

#### Encryption Key (Required for Token Storage)

```bash
ENCRYPTION_KEY=your_32_character_base64_key
```

**Generating an Encryption Key:**

```bash
# Generate a secure Fernet key
python3 -c "from cryptography.fernet import Fernet; print('ENCRYPTION_KEY=' + Fernet.generate_key().decode())"
```

Or copy this example (generate your own for production):
```bash
ENCRYPTION_KEY=xspgAnxDFrRIrKQ0ZhywRV4hcAIYmjFQ7Aavn7EHt6M=
```

**⚠️ Security Notes:**
- This key is used to encrypt sensitive tokens stored locally
- Generate a unique key for each environment (development, production)
- Never commit the actual `.env` file to version control
- The `.env` file is automatically ignored by git for security

### Complete .env Example

```bash
# Data directory (optional)
DATA_DIR=./data

# Strava API Configuration
STRAVA_CLIENT_ID=12345
STRAVA_CLIENT_SECRET=abcdef123456789
STRAVA_REDIRECT_URI=http://localhost:8501/callback

# Encryption key for secure token storage
ENCRYPTION_KEY=xspgAnxDFrRIrKQ0ZhywRV4hcAIYmjFQ7Aavn7EHt6M=

# Mapbox (optional, required for premium basemaps in Activity map)
MAPBOX_TOKEN=pk.your_mapbox_access_token_here
```

#### Mapbox Basemap (Optional)

To unlock additional background maps (Satellite, Outdoors, Light/Dark variations) on the Activity detail page, add a Mapbox access token to your `.env`:

```bash
MAPBOX_TOKEN=pk.your_mapbox_access_token_here
```

You can generate a token from the [Mapbox account dashboard](https://account.mapbox.com/access-tokens/).  
If this value is omitted, the Activity map will fall back to the default OpenStreetMap style and the Mapbox options will be hidden.

## 🔐 Security Considerations

### Environment Variables Security

- ✅ **`.env` files are git-ignored** - Your secrets won't be committed
- ✅ **Fernet encryption** - Tokens are encrypted before storage
- ✅ **No hardcoded secrets** - All sensitive data comes from environment
- ❌ **Never log secrets** - Use the `redact()` utility for logging
- ❌ **Don't share `.env` files** - Each environment should have unique keys

### API Security Best Practices

1. **Use HTTPS in production** for all API communications
2. **Rotate encryption keys** periodically in production
3. **Monitor token usage** and implement rate limiting
4. **Store tokens encrypted** (handled automatically by the app)

## 🧪 Testing Your Setup

### Verify Environment Loading

```bash
# Test that your .env file is loaded correctly
python3 -c "
from utils.config import load_config, redact
config = load_config()
print('Configuration loaded successfully!')
print(f'Data dir: {config.data_dir}')
print(f'Strava Client ID: {redact(config.strava_client_id)}')
print(f'Encryption key configured: {bool(config.encryption_key)}')
"
```

### Test Strava Integration

1. Start the application: `uv run streamlit run app.py`
2. Navigate to Settings page
3. Check that all configuration values are detected
4. Test Strava OAuth flow if credentials are configured

## 🚨 Troubleshooting

### Common Issues

**"ENCRYPTION_KEY is required for token storage"**
- Generate an encryption key using the command above
- Ensure the key is exactly copied to your `.env` file

**"STRAVA_CLIENT_ID and STRAVA_REDIRECT_URI must be configured"**
- Verify your Strava API credentials in the Strava dashboard
- Ensure the redirect URI matches exactly (case-sensitive)

**Environment variables not loading**
- Check that `.env` exists in the project root
- Verify the file format and that values are not commented out
- Restart your Python environment after changes

### Getting Help

- Check the application logs for detailed error messages
- Ensure all dependencies are installed: `uv sync`
- Verify Python version: `python3 --version` (should be 3.11+)

## 📚 Additional Resources

- [Strava API Documentation](https://developers.strava.com/docs/)
- [Python cryptography library](https://cryptography.io/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [uv Package Manager](https://github.com/astral-sh/uv)
