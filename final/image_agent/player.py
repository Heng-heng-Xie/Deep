import numpy as np
import torch
import torchvision
import time
from PIL import Image
from os import path
from .models import load_model



if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')



class Team:
    agent_type = 'image'

    def __init__(self):
        """
          TODO: Load your agent here. Load network parameters, and other parts of our model
          We will call this function with default arguments only
        """
        self.goals = np.float32([[0, 75], [0, -75]])
        self.kart = 'wilber'
        self.init()
        self.model = load_model(path.join(path.dirname(path.abspath(__file__)), 'det.th')).to(device)
        self.model.eval()
        self.transform = torchvision.transforms.Compose([torchvision.transforms.Resize((128, 128)),
                                                         torchvision.transforms.ToTensor()])

    def new_match(self, team: int, num_players: int) -> list:
        """
        Let's start a new match. You're playing on a `team` with `num_players` and have the option of choosing your kart
        type (name) for each player.
        :param team: What team are you playing on RED=0 or BLUE=1
        :param num_players: How many players are there on your team
        :return: A list of kart names. Choose from 'adiumy', 'amanda', 'beastie', 'emule', 'gavroche', 'gnu', 'hexley',
                 'kiki', 'konqi', 'nolok', 'pidgin', 'puffy', 'sara_the_racer', 'sara_the_wizard', 'suzanne', 'tux',
                 'wilber', 'xue'. Default: 'tux'
        """
        """
           TODO: feel free to edit or delete any of the code below
        """
        self.team, self.num_players = team, num_players

        self.init()
        print(f"Using {device} Match Started: {time.strftime('%H-%M-%S')}")
        return [self.kart] * num_players



    def act(self, player_state, player_image):
        """
        This function is called once per timestep. You're given a list of player_states and images.

        DO NOT CALL any pystk functions here. It will crash your program on your grader.

        :param player_state: list[dict] describing the state of the players of this team. The state closely follows
                             the pystk.Player object <https://pystk.readthedocs.io/en/latest/state.html#pystk.Player>.
                             See HW5 for some inspiration on how to use the camera information.
                             camera:  Camera info for each player
                               - aspect:     Aspect ratio
                               - fov:        Field of view of the camera
                               - mode:       Most likely NORMAL (0)
                               - projection: float 4x4 projection matrix
                               - view:       float 4x4 view matrix
                             kart:  Information about the kart itself
                               - front:     float3 vector pointing to the front of the kart
                               - location:  float3 location of the kart
                               - rotation:  float4 (quaternion) describing the orientation of kart (use front instead)
                               - size:      float3 dimensions of the kart
                               - velocity:  float3 velocity of the kart in 3D

        :param player_image: list[np.array] showing the rendered image from the viewpoint of each kart. Use
                             player_state[i]['camera']['view'] and player_state[i]['camera']['projection'] to find out
                             from where the image was taken.

        :return: dict  The action to be taken as a dictionary. For example `dict(acceleration=1, steer=0.25)`.
                 acceleration: float 0..1
                 brake:        bool Brake will reverse if you do not accelerate (good for backing up)
                 drift:        bool (optional. unless you want to turn faster)
                 fire:         bool (optional. you can hit the puck with a projectile)
                 nitro:        bool (optional)
                 rescue:       bool (optional. no clue where you will end up though.)
                 steer:        float -1..1 steering angle
        """
        # TODO: Change me. I'm just cruising straight

        # player1 action
        player1_info = player_state[0]
        image1 = player_image[0]


        if np.linalg.norm(player1_info['kart']['velocity']) < 1:
            if self.timer == 0:
                self.timer = self.step
            elif self.step - self.timer > 20:
                self.init()
        else:
            self.timer = 0
        player1_act = self._playing(player1_info, image1)

        #player2 action
        player2_info = player_state[1]
        image2 = player_image[1]
        player2_act = self._playing(player2_info, image2)
        self.step += 1
        return [player1_act, player2_act]


    def _playing(self, player_info, image):
        def goal_dist_angel(front, loc_kart, team):
            ori = front - loc_kart
            ori = ori / np.linalg.norm(ori)
            G_ori = self.goals[team] - loc_kart
            G_dist = np.linalg.norm(G_ori)
            G_ori = G_ori / np.linalg.norm(G_ori)
            angle = np.arccos(np.clip(np.dot(ori, G_ori), -1, 1))
            degree = np.degrees(-np.sign(np.cross(ori, G_ori)) * angle)
            return G_dist, degree

        img = self.transform(Image.fromarray(image)).to(device)
        pred = self.model.detect(
            img, max_pool_ks=7, min_score=0.2, max_det=15)

        front = np.float32(player_info['kart']['front'])[[0, 2]]
        loc_kart = np.float32(player_info['kart']['location'])[[0, 2]]

        puck_det = len(pred) > 0
        if puck_det:
            puck_loc = np.mean([cx[1] for cx in pred])
            puck_loc = puck_loc / 64 - 1

            if self.puck_kick and np.abs(puck_loc - self.puck_last) > 0.7:
                puck_loc = self.puck_last
                self.puck_kick = False
            else:
                self.puck_kick = True

            self.puck_last = puck_loc
            self.last_found = self.step

        elif self.step - self.last_found < 4:
            self.puck_kick = False
            puck_loc = self.puck_last
        else:
            puck_loc = None
            self.step_back = 10

        G_self_dist, G_self_deg = goal_dist_angel(front, loc_kart, self.team-1)

        G_enemy_dist, G_enemy_deg = goal_dist_angel(front, loc_kart, self.team)

        G_enemy_dist = ((np.clip(G_enemy_dist, 10, 100) - 10) / 90) + 1
        
        if self.step_back == 0 and (self.cooldown_lost == 0 or puck_det):
            if 20 < np.abs(G_enemy_deg) < 120:
                distW = 1 / G_enemy_dist ** 3
                aim_point = puck_loc + \
                    np.sign(puck_loc - G_enemy_deg /
                            100) * 0.3 * distW
            else:
                aim_point = puck_loc
            if self.last_found == self.step:
                brake = False
                acceleration = 0.75 if np.linalg.norm(
                    player_info['kart']['velocity']) < 15 else 0
            else:
                acceleration = 0
                brake = False
        elif self.cooldown_lost > 0:
            self.cooldown_lost -= 1
            brake = False
            acceleration = 0.5
            aim_point = G_enemy_deg / 100
        else:
            if G_self_dist > 10:
                acceleration = 0
                brake = True
                aim_point = G_self_deg / 100
                self.step_back -= 1
            else:
                self.cooldown_lost = 10
                self.cooldown_lost = 0
                aim_point = G_enemy_deg / 100
                acceleration = 0.5
                brake = False

        steer = np.clip(aim_point * 15, -1, 1)
        drift = np.abs(aim_point) > 0.7

        player_action = {
            'steer': G_enemy_deg if self.step < 25 else steer,
            'acceleration': 1 if self.step < 25 else acceleration,
            'brake': brake,
            'drift': drift,
            'nitro': False,
            'rescue': False
        }
        return player_action


    def init(self):
        self.step = 0
        self.timer = 0

        self.puck_last = 0
        self.last_found = 0
        self.step_back = 0
        self.puck_kick = True
        self.cooldown_lost = 0
