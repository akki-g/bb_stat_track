# stats_engine.py
import json
import time
from datetime import datetime
from collections import defaultdict

class StatsEngine:
    def __init__(self):
        self.game_stats = {
            'start_time': datetime.now().isoformat(),
            'teams': {
                'team_a': {'players': {}, 'total': defaultdict(int)},
                'team_b': {'players': {}, 'total': defaultdict(int)}
            },
            'events': []
        }
        
        self.player_assignments = {}  # track_id -> team/player mapping
        
    def assign_player(self, track_id, team, player_name):
        """Assign a tracked player to a team"""
        self.player_assignments[track_id] = {
            'team': team,
            'name': player_name
        }
        
        # Initialize player stats
        if player_name not in self.game_stats['teams'][team]['players']:
            self.game_stats['teams'][team]['players'][player_name] = {
                'shots_attempted': 0,
                'shots_made': 0,
                'turnovers': 0,
                'travels': 0,
                'assists': 0,
                'steals': 0,
                'fouls': 0,
                'playing_time': 0
            }
    
    def record_shot(self, track_id, made, frame_number, court_position):
        """Record a shot attempt"""
        player_info = self.player_assignments.get(track_id)
        if not player_info:
            return
        
        team = player_info['team']
        player = player_info['name']
        
        # Update stats
        self.game_stats['teams'][team]['players'][player]['shots_attempted'] += 1
        if made:
            self.game_stats['teams'][team]['players'][player]['shots_made'] += 1
        
        # Record event
        event = {
            'type': 'shot',
            'frame': frame_number,
            'time': datetime.now().isoformat(),
            'player': player,
            'team': team,
            'made': made,
            'court_position': court_position
        }
        self.game_stats['events'].append(event)
        
        # Update team totals
        self.game_stats['teams'][team]['total']['shots_attempted'] += 1
        if made:
            self.game_stats['teams'][team]['total']['shots_made'] += 1
    
    def record_action(self, track_id, action_type, frame_number):
        """Record various game actions"""
        player_info = self.player_assignments.get(track_id)
        if not player_info:
            return
        
        team = player_info['team']
        player = player_info['name']
        
        # Map action types to stat categories
        stat_mapping = {
            'travel': 'travels',
            'turnover': 'turnovers',
            'steal': 'steals',
            'foul': 'fouls'
        }
        
        if action_type in stat_mapping:
            stat_key = stat_mapping[action_type]
            self.game_stats['teams'][team]['players'][player][stat_key] += 1
            self.game_stats['teams'][team]['total'][stat_key] += 1
        
        # Record event
        event = {
            'type': action_type,
            'frame': frame_number,
            'time': datetime.now().isoformat(),
            'player': player,
            'team': team
        }
        self.game_stats['events'].append(event)
    
    def get_player_stats(self, team, player):
        """Get current stats for a player"""
        return self.game_stats['teams'][team]['players'].get(player, {})
    
    def get_team_stats(self, team):
        """Get aggregated team statistics"""
        return dict(self.game_stats['teams'][team]['total'])
    
    def save_stats(self, filename='game_stats.json'):
        """Save statistics to file"""
        # Convert defaultdict instances to regular dicts for JSON serialization
        serializable = {
            'start_time': self.game_stats['start_time'],
            'teams': {},
            'events': self.game_stats['events']
        }

        for team, data in self.game_stats['teams'].items():
            serializable['teams'][team] = {
                'players': data['players'],
                'total': dict(data['total'])
            }

        with open(filename, 'w') as f:
            json.dump(serializable, f, indent=2)
    
    def generate_summary(self):
        """Generate game summary"""
        summary = []
        summary.append("=== GAME STATISTICS ===\n")
        
        for team in ['team_a', 'team_b']:
            summary.append(f"\n{team.upper()} Stats:")
            team_data = self.game_stats['teams'][team]
            
            # Team totals
            totals = team_data['total']
            if totals['shots_attempted'] > 0:
                fg_pct = (totals['shots_made'] / totals['shots_attempted']) * 100
                summary.append(f"  Shooting: {totals['shots_made']}/{totals['shots_attempted']} ({fg_pct:.1f}%)")
            
            summary.append(f"  Turnovers: {totals.get('turnovers', 0)}")
            summary.append(f"  Travels: {totals.get('travels', 0)}")
            
            # Individual players
            summary.append("\n  Player Stats:")
            for player, stats in team_data['players'].items():
                summary.append(f"    {player}:")
                if stats['shots_attempted'] > 0:
                    fg_pct = (stats['shots_made'] / stats['shots_attempted']) * 100
                    summary.append(f"      Shooting: {stats['shots_made']}/{stats['shots_attempted']} ({fg_pct:.1f}%)")
                summary.append(f"      Turnovers: {stats['turnovers']}")
                summary.append(f"      Travels: {stats['travels']}")
        
        return '\n'.join(summary)
    

    