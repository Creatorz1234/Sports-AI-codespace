# Sports AI Analyzer in Codespace

This project contains a `SportsAIAnalyzer` class for predicting sports game outcomes using SportsRadar and OpenWeatherMap APIs.

## Setup Instructions

1. **Open in Codespace**:
   - If using GitHub Codespaces, click "Code" > "Open with Codespaces" in your repository.
   - The `.devcontainer/devcontainer.json` will automatically set up the environment.

2. **Install Dependencies** (if not using Codespaces auto-setup):
   - Run the following command in the terminal:
     ```
     pip install -r requirements.txt
     ```

3. **Add API Keys**:
   - Open `config.yaml` and replace the placeholder API keys:
     ```yaml
     sportsradar_api_key: "your_trial_sportsradar_key"
     weather_api_key: "your_openweathermap_key"
     ```

4. **Adjust Game/Player IDs**:
   - In `sports_ai_analyzer.py`, replace `"player1"` and `"12345"` with real player and game IDs from SportsRadar.
   - Use the schedule endpoint (e.g., `/nba/trial/v8/en/schedules/2025-04-07/schedule.json`) to find game IDs.

5. **Run the Code**:
   - Execute the script:
     ```
     python sports_ai_analyzer.py
     ```

## Notes
- **Quota Limits**: The SportsRadar trial has a 1,000 requests/month limit and 1 request/second rate limit.
- **Supported Sports**: Focus on NBA, MLB, WNBA, NHL, and Soccer, as NASCAR, MMA, Tennis, and Golf may have limited trial support.
- **Logs**: Check `sports_ai.log` for errors or API issues.
