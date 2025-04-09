import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
import logging
import requests
import json
import time
import yaml
from cachetools import TTLCache
from typing import Dict, Optional

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='sports_ai.log'
)

class SportsAIAnalyzer:
    def __init__(self, sport: str = "basketball", config_path: str = "config.yaml"):
        self.sport = sport.lower()
        self.config = self._load_config(config_path)
        self.player_data: Dict = {}
        self.team_data: Dict = {}
        self.game_history: pd.DataFrame = pd.DataFrame()
        self.model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        self.scaler = StandardScaler()
        
        self.sportsradar_api_key = self.config.get("sportsradar_api_key", "SPORTSRADAR_API_KEY")
        self.weather_api_key = self.config.get("weather_api_key", "WEATHER_API_KEY")
        
        self.sport_prefix = {
            "basketball": "nba", "baseball": "mlb", "soccer": "soccer",
            "hockey": "nhl", "nascar": "nascar", "mma": "mma",
            "tennis": "tennis", "golf": "golf", "wnba": "wnba"
        }.get(self.sport, "nba")
        
        self.api_version = {
            "mlb": "v8", "nba": "v8", "wnba": "v8", "nhl": "v8",
            "nascar": "v3", "mma": "v2", "soccer": "v4", "tennis": "v3", "golf": "v2"
        }.get(self.sport_prefix, "v8")
        
        self.cache = TTLCache(maxsize=1000, ttl=3600)
        self.last_request_time = 0  # For rate limiting

    def _load_config(self, config_path: str) -> Dict:
        default_config = {
            "sportsradar_api_key": "SPORTSRADAR_API_KEY",
            "weather_api_key": "WEATHER_API_KEY",
            "stats_validation_rules": {
                "basketball": {"points": {"min": 0, "max": 100, "type": "int"}},
                "baseball": {"strikeouts": {"min": 0, "max": 20, "type": "int"}},
                "soccer": {"goals": {"min": 0, "max": 10, "type": "int"}},
                "hockey": {"goals": {"min": 0, "max": 20, "type": "int"}},
                "nascar": {"position": {"min": 1, "max": 40, "type": "int"}},
                "mma": {"strikes": {"min": 0, "max": 200, "type": "int"}},
                "tennis": {"aces": {"min": 0, "max": 50, "type": "int"}},
                "golf": {"strokes": {"min": 0, "max": 100, "type": "int"}},
                "wnba": {"points": {"min": 0, "max": 100, "type": "int"}}
            },
            "external_factors": {"wind_impact": 0.1}
        }
        try:
            with open(config_path, 'r') as f:
                file_config = yaml.safe_load(f) or {}
            default_config.update(file_config)
        except Exception as e:
            logging.warning(f"Error loading config: {str(e)}. Using defaults")
        return default_config

    def _validate_stats(self, stats: Dict) -> bool:
        try:
            rules = self.config["stats_validation_rules"].get(self.sport, {})
            for stat, value in stats.items():
                if stat in rules and not (rules[stat]["min"] <= value <= rules[stat]["max"]):
                    raise ValueError(f"Stat {stat} out of range")
            return True
        except Exception as e:
            logging.error(f"Stats validation error: {str(e)}")
            return False

    def _rate_limit(self):
        """Enforce 1 request/second limit"""
        current_time = time.time()
        if current_time - self.last_request_time < 1:
            time.sleep(1 - (current_time - self.last_request_time))
        self.last_request_time = time.time()

    def fetch_injury_data(self, player_id: str) -> Dict:
        cache_key = f"injury_{player_id}"
        if cache_key in self.cache:
            return self.cache[cache_key]import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
import logging
import requests
import json
import time
import yaml
from cachetools import TTLCache
from typing import Dict, Optional

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='sports_ai.log'
)

class SportsAIAnalyzer:
    def __init__(self, sport: str = "basketball", config_path: str = "config.yaml"):
        self.sport = sport.lower()
        self.config = self._load_config(config_path)
        self.player_data: Dict = {}
        self.team_data: Dict = {}
        self.game_history: pd.DataFrame = pd.DataFrame()
        self.model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        self.scaler = StandardScaler()
        
        self.sportsradar_api_key = self.config.get("sportsradar_api_key", "SPORTSRADAR_API_KEY")
        self.weather_api_key = self.config.get("weather_api_key", "WEATHER_API_KEY")
        
        self.sport_prefix = {
            "basketball": "nba", "baseball": "mlb", "soccer": "soccer",
            "hockey": "nhl", "nascar": "nascar", "mma": "mma",
            "tennis": "tennis", "golf": "golf", "wnba": "wnba"
        }.get(self.sport, "nba")
        
        self.api_version = {
            "mlb": "v8", "nba": "v8", "wnba": "v8", "nhl": "v8",
            "nascar": "v3", "mma": "v2", "soccer": "v4", "tennis": "v3", "golf": "v2"
        }.get(self.sport_prefix, "v8")
        
        self.cache = TTLCache(maxsize=1000, ttl=3600)
        self.last_request_time = 0  # For rate limiting

    def _load_config(self, config_path: str) -> Dict:
        default_config = {
            "sportsradar_api_key": "SPORTSRADAR_API_KEY",
            "weather_api_key": "WEATHER_API_KEY",
            "stats_validation_rules": {
                "basketball": {"points": {"min": 0, "max": 100, "type": "int"}},
                "baseball": {"strikeouts": {"min": 0, "max": 20, "type": "int"}},
                "soccer": {"goals": {"min": 0, "max": 10, "type": "int"}},
                "hockey": {"goals": {"min": 0, "max": 20, "type": "int"}},
                "nascar": {"position": {"min": 1, "max": 40, "type": "int"}},
                "mma": {"strikes": {"min": 0, "max": 200, "type": "int"}},
                "tennis": {"aces": {"min": 0, "max": 50, "type": "int"}},
                "golf": {"strokes": {"min": 0, "max": 100, "type": "int"}},
                "wnba": {"points": {"min": 0, "max": 100, "type": "int"}}
            },
            "external_factors": {"wind_impact": 0.1}
        }
        try:
            with open(config_path, 'r') as f:
                file_config = yaml.safe_load(f) or {}
            default_config.update(file_config)
        except Exception as e:
            logging.warning(f"Error loading config: {str(e)}. Using defaults")
        return default_config

    def _validate_stats(self, stats: Dict) -> bool:
        try:
            rules = self.config["stats_validation_rules"].get(self.sport, {})
            for stat, value in stats.items():
                if stat in rules and not (rules[stat]["min"] <= value <= rules[stat]["max"]):
                    raise ValueError(f"Stat {stat} out of range")
            return True
        except Exception as e:
            logging.error(f"Stats validation error: {str(e)}")
            return False

    def _rate_limit(self):
        """Enforce 1 request/second limit"""
        current_time = time.time()
        if current_time - self.last_request_time < 1:
            time.sleep(1 - (current_time - self.last_request_time))
        self.last_request_time = time.time()

    def fetch_injury_data(self, player_id: str) -> Dict:
        cache_key = f"injury_{player_id}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            self._rate_limit()
            url = f"https://api.sportsdata.io/{self.api_version}/{self.sport_prefix}/injuries.json?key={self.sportsradar_api_key}"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            injuries = response.json()
            injury_data = next((item for item in injuries if str(item["PlayerID"]) == player_id), 
                             {"Status": "Healthy", "Injury": None, "Updated": None})
            result = {
                "injury_status": injury_data["Status"],
                "injury_type": injury_data.get("Injury"),
                "last_updated": injury_data.get("Updated")
            }
            self.cache[cache_key] = result
            return result
        except requests.RequestException as e:
            logging.error(f"Error fetching injury data: {str(e)}")
            return {"injury_status": "Unknown", "injury_type": None, "last_updated": None}

    def fetch_weather_data(self, venue: str) -> Dict:
        cache_key = f"weather_{venue}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            url = f"http://api.openweathermap.org/data/2.5/weather?q={venue}&appid={self.weather_api_key}&units=imperial"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            weather = response.json()
            weather_data = {
                "temperature": weather["main"]["temp"],
                "wind_speed": weather["wind"]["speed"]
            }
            self.cache[cache_key] = weather_data
            return weather_data
        except requests.RequestException as e:
            logging.error(f"Error fetching weather data: {str(e)}")
            return {"temperature": 0.0, "wind_speed": 0.0}

    def fetch_prematch_odds(self, game_id: str) -> Dict:
        cache_key = f"prematch_odds_{game_id}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            self._rate_limit()
            url = f"https://api.sportsdata.io/oddscomparison/prematch/v1/en/sports/{self.sport_prefix}/matches.json?key={self.sportsradar_api_key}"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            matches = response.json()
            odds_data = next((match for match in matches if match.get("id") == game_id), None)
            result = {
                "home_odds": odds_data["markets"][0]["outcomes"][0]["odds"] if odds_data else 0.0,
                "away_odds": odds_data["markets"][0]["outcomes"][1]["odds"] if odds_data else 0.0
            } if odds_data else {"home_odds": 0.0, "away_odds": 0.0}
            self.cache[cache_key] = result
            return result
        except requests.RequestException as e:
            logging.error(f"Error fetching prematch odds: {str(e)}")
            return {"home_odds": 0.0, "away_odds": 0.0}

    def fetch_player_props(self, player_id: str, game_id: str) -> Dict:
        cache_key = f"player_props_{player_id}_{game_id}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            self._rate_limit()
            url = f"https://api.sportsdata.io/oddscomparison/playerprops/v1/en/sports/{self.sport_prefix}/propositions.json?key={self.sportsradar_api_key}"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            props = response.json()
            player_prop = next((prop for prop in props if prop.get("player_id") == player_id and prop.get("game_id") == game_id), None)
            result = {
                "prop_odds": player_prop["odds"] if player_prop else 0.0
            } if player_prop else {"prop_odds": 0.0}
            self.cache[cache_key] = result
            return result
        except requests.RequestException as e:
            logging.error(f"Error fetching player props: {str(e)}")
            return {"prop_odds": 0.0}

    def collect_player_data(self, player_id: str, stats: Dict, game_id: str = None) -> bool:
        try:
            if not self._validate_stats(stats):
                raise ValueError("Invalid stats")
            if player_id not in self.player_data:
                self.player_data[player_id] = {'stats': [], 'injury_status': None, 'prop_odds': 0.0}
            injury_data = self.fetch_injury_data(player_id)
            player_props = self.fetch_player_props(player_id, game_id) if game_id else {"prop_odds": 0.0}
            self.player_data[player_id]['stats'].append({'stats': stats})
            self.player_data[player_id]['injury_status'] = injury_data["injury_status"]
            self.player_data[player_id]['prop_odds'] = player_props["prop_odds"]
            return True
        except Exception as e:
            logging.error(f"Error collecting player data: {str(e)}")
            return False

    def update_game_history(self, game_data: Dict) -> bool:
        try:
            game_id = game_data.get("game_id", "unknown")
            weather_data = self.fetch_weather_data(game_data.get('venue', "unknown"))
            prematch_odds = self.fetch_prematch_odds(game_id)
            game_conditions = game_data.get('conditions', {})
            game_conditions.update(weather_data)
            
            game_record = {
                'home_team': game_data['home_team'],
                'away_team': game_data['away_team'],
                'result': game_data.get('result'),
                'features': json.dumps(self._prepare_game_features(
                    game_data['home_team'], game_data['away_team'], game_conditions, prematch_odds
                ))
            }
            self.game_history = pd.concat([self.game_history, pd.DataFrame([game_record])], ignore_index=True)
            return True
        except Exception as e:
            logging.error(f"Error updating game history: {str(e)}")
            return False

    def _prepare_game_features(self, team1_id: str, team2_id: str, game_conditions: Dict, prematch_odds: Dict) -> list:
        try:
            team1 = self.team_data.get(team1_id, {})
            team2 = self.team_data.get(team2_id, {})
            features = [
                team1.get('current_form', 0.0),
                team2.get('current_form', 0.0),
                game_conditions.get('wind_speed', 0.0) * self.config["external_factors"]["wind_impact"],
                game_conditions.get('temperature', 0.0),
                prematch_odds["home_odds"],
                prematch_odds["away_odds"],
                max([p.get('prop_odds', 0.0) for p in self.player_data.values()], default=0.0)
            ]
            return features
        except Exception as e:
            logging.error(f"Error preparing features: {str(e)}")
            return [0.0] * 7import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
import logging
import requests
import json
import time
import yaml
from cachetools import TTLCache
from typing import Dict, Optional

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='sports_ai.log'
)

class SportsAIAnalyzer:
    def __init__(self, sport: str = "basketball", config_path: str = "config.yaml"):
        self.sport = sport.lower()
        self.config = self._load_config(config_path)
        self.player_data: Dict = {}
        self.team_data: Dict = {}
        self.game_history: pd.DataFrame = pd.DataFrame()
        self.model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        self.scaler = StandardScaler()
        
        self.sportsradar_api_key = self.config.get("sportsradar_api_key", "SPORTSRADAR_API_KEY")
        self.weather_api_key = self.config.get("weather_api_key", "WEATHER_API_KEY")
        
        self.sport_prefix = {
            "basketball": "nba", "baseball": "mlb", "soccer": "soccer",
            "hockey": "nhl", "nascar": "nascar", "mma": "mma",
            "tennis": "tennis", "golf": "golf", "wnba": "wnba"
        }.get(self.sport, "nba")
        
        self.api_version = {
            "mlb": "v8", "nba": "v8", "wnba": "v8", "nhl": "v8",
            "nascar": "v3", "mma": "v2", "soccer": "v4", "tennis": "v3", "golf": "v2"
        }.get(self.sport_prefix, "v8")
        
        self.cache = TTLCache(maxsize=1000, ttl=3600)
        self.last_request_time = 0  # For rate limiting

    def _load_config(self, config_path: str) -> Dict:
        default_config = {
            "sportsradar_api_key": "SPORTSRADAR_API_KEY",
            "weather_api_key": "WEATHER_API_KEY",
            "stats_validation_rules": {
                "basketball": {"points": {"min": 0, "max": 100, "type": "int"}},
                "baseball": {"strikeouts": {"min": 0, "max": 20, "type": "int"}},
                "soccer": {"goals": {"min": 0, "max": 10, "type": "int"}},
                "hockey": {"goals": {"min": 0, "max": 20, "type": "int"}},
                "nascar": {"position": {"min": 1, "max": 40, "type": "int"}},
                "mma": {"strikes": {"min": 0, "max": 200, "type": "int"}},
                "tennis": {"aces": {"min": 0, "max": 50, "type": "int"}},
                "golf": {"strokes": {"min": 0, "max": 100, "type": "int"}},
                "wnba": {"points": {"min": 0, "max": 100, "type": "int"}}
            },
            "external_factors": {"wind_impact": 0.1}
        }
        try:
            with open(config_path, 'r') as f:
                file_config = yaml.safe_load(f) or {}
            default_config.update(file_config)
        except Exception as e:
            logging.warning(f"Error loading config: {str(e)}. Using defaults")
        return default_config

    def _validate_stats(self, stats: Dict) -> bool:
        try:
            rules = self.config["stats_validation_rules"].get(self.sport, {})
            for stat, value in stats.items():
                if stat in rules and not (rules[stat]["min"] <= value <= rules[stat]["max"]):
                    raise ValueError(f"Stat {stat} out of range")
            return True
        except Exception as e:
            logging.error(f"Stats validation error: {str(e)}")
            return False

    def _rate_limit(self):
        """Enforce 1 request/second limit"""
        current_time = time.time()
        if current_time - self.last_request_time < 1:
            time.sleep(1 - (current_time - self.last_request_time))
        self.last_request_time = time.time()

    def fetch_injury_data(self, player_id: str) -> Dict:
        cache_key = f"injury_{player_id}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            self._rate_limit()
            url = f"https://api.sportsdata.io/{self.api_version}/{self.sport_prefix}/injuries.json?key={self.sportsradar_api_key}"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            injuries = response.json()
            injury_data = next((item for item in injuries if str(item["PlayerID"]) == player_id), 
                             {"Status": "Healthy", "Injury": None, "Updated": None})
            result = {
                "injury_status": injury_data["Status"],
                "injury_type": injury_data.get("Injury"),
                "last_updated": injury_data.get("Updated")
            }
            self.cache[cache_key] = result
            return result
        except requests.RequestException as e:
            logging.error(f"Error fetching injury data: {str(e)}")
            return {"injury_status": "Unknown", "injury_type": None, "last_updated": None}

    def fetch_weather_data(self, venue: str) -> Dict:
        cache_key = f"weather_{venue}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            url = f"http://api.openweathermap.org/data/2.5/weather?q={venue}&appid={self.weather_api_key}&units=imperial"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            weather = response.json()
            weather_data = {
                "temperature": weather["main"]["temp"],
                "wind_speed": weather["wind"]["speed"]
            }
            self.cache[cache_key] = weather_data
            return weather_data
        except requests.RequestException as e:
            logging.error(f"Error fetching weather data: {str(e)}")
            return {"temperature": 0.0, "wind_speed": 0.0}

    def fetch_prematch_odds(self, game_id: str) -> Dict:
        cache_key = f"prematch_odds_{game_id}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            self._rate_limit()
            url = f"https://api.sportsdata.io/oddscomparison/prematch/v1/en/sports/{self.sport_prefix}/matches.json?key={self.sportsradar_api_key}"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            matches = response.json()
            odds_data = next((match for match in matches if match.get("id") == game_id), None)
            result = {
                "home_odds": odds_data["markets"][0]["outcomes"][0]["odds"] if odds_data else 0.0,
                "away_odds": odds_data["markets"][0]["outcomes"][1]["odds"] if odds_data else 0.0
            } if odds_data else {"home_odds": 0.0, "away_odds": 0.0}
            self.cache[cache_key] = result
            return result
        except requests.RequestException as e:
            logging.error(f"Error fetching prematch odds: {str(e)}")
            return {"home_odds": 0.0, "away_odds": 0.0}

    def fetch_player_props(self, player_id: str, game_id: str) -> Dict:
        cache_key = f"player_props_{player_id}_{game_id}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            self._rate_limit()
            url = f"https://api.sportsdata.io/oddscomparison/playerprops/v1/en/sports/{self.sport_prefix}/propositions.json?key={self.sportsradar_api_key}"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            props = response.json()
            player_prop = next((prop for prop in props if prop.get("player_id") == player_id and prop.get("game_id") == game_id), None)
            result = {
                "prop_odds": player_prop["odds"] if player_prop else 0.0
            } if player_prop else {"prop_odds": 0.0}
            self.cache[cache_key] = result
            return result
        except requests.RequestException as e:
            logging.error(f"Error fetching player props: {str(e)}")
            return {"prop_odds": 0.0}

    def collect_player_data(self, player_id: str, stats: Dict, game_id: str = None) -> bool:
        try:
            if not self._validate_stats(stats):
                raise ValueError("Invalid stats")
            if player_id not in self.player_data:
                self.player_data[player_id] = {'stats': [], 'injury_status': None, 'prop_odds': 0.0}
            injury_data = self.fetch_injury_data(player_id)
            player_props = self.fetch_player_props(player_id, game_id) if game_id else {"prop_odds": 0.0}
            self.player_data[player_id]['stats'].append({'stats': stats})
            self.player_data[player_id]['injury_status'] = injury_data["injury_status"]
            self.player_data[player_id]['prop_odds'] = player_props["prop_odds"]
            return True
        except Exception as e:
            logging.error(f"Error collecting player data: {str(e)}")
            return False

    def update_game_history(self, game_data: Dict) -> bool:
        try:
            game_id = game_data.get("game_id", "unknown")
            weather_data = self.fetch_weather_data(game_data.get('venue', "unknown"))
            prematch_odds = self.fetch_prematch_odds(game_id)
            game_conditions = game_data.get('conditions', {})
            game_conditions.update(weather_data)
            
            game_record = {
                'home_team': game_data['home_team'],
                'away_team': game_data['away_team'],
                'result': game_data.get('result'),
                'features': json.dumps(self._prepare_game_features(
                    game_data['home_team'], game_data['away_team'], game_conditions, prematch_odds
                ))
            }
            self.game_history = pd.concat([self.game_history, pd.DataFrame([game_record])], ignore_index=True)
            return True
        except Exception as e:
            logging.error(f"Error updating game history: {str(e)}")
            return False

    def _prepare_game_features(self, team1_id: str, team2_id: str, game_conditions: Dict, prematch_odds: Dict) -> list:
        try:
            team1 = self.team_data.get(team1_id, {})
            team2 = self.team_data.get(team2_id, {})
            features = [
                team1.get('current_form', 0.0),
                team2.get('current_form', 0.0),
                game_conditions.get('wind_speed', 0.0) * self.config["external_factors"]["wind_impact"],
                game_conditions.get('temperature', 0.0),
                prematch_odds["home_odds"],
                prematch_odds["away_odds"],
                max([p.get('prop_odds', 0.0) for p in self.player_data.values()], default=0.0)
            ]
            return features
        except Exception as e:
            logging.error(f"Error preparing features: {str(e)}")
            return [0.0] * 7

    def predict_game_outcome(self, team1_id: str, team2_id: str, game_conditions: Dict) -> Optional[Dict]:
        try:
            weather_data = self.fetch_weather_data(game_conditions.get('venue', "unknown"))
            game_conditions.update(weather_data)
            game_id = game_conditions.get("game_id", "unknown")
            prematch_odds = self.fetch_prematch_odds(game_id)
            features = self._prepare_game_features(team1_id, team2_id, game_conditions, prematch_odds)
            features_scaled = self.scaler.fit_transform([features])  # Simplified for demo
            prediction = self.model.fit(features_scaled, [1]).predict(features_scaled)[0]  # Dummy training for demo
            probability = self.model.predict_proba(features_scaled)[0]
            return {
                'predicted_winner': team1_id if prediction == 1 else team2_id,
                'confidence': float(max(probability))
            }
        except Exception as e:
            logging.error(f"Error predicting outcome: {str(e)}")
            return None

def main():
    sports_ai = SportsAIAnalyzer(sport="basketball")
    player_stats = {"points": 25}
    sports_ai.collect_player_data("player1", player_stats, game_id="12345")
    game_data = {"home_team": "team1", "away_team": "team2", "result": 1, 
                 "conditions": {"venue": "Boston"}, "game_id": "12345"}
    sports_ai.update_game_history(game_data)
    prediction = sports_ai.predict_game_outcome("team1", "team2", game_data["conditions"])
    print(prediction)

if __name__ == "__main__":
    main()
