import vizdoom as vzd
import torch
import torchvision.transforms as T

class DoomEnv:
    def __init__(self, cfg):
        self.game_ = vzd.DoomGame()
        self.game_.load_config(cfg)
        self.game_.init()

        self.state_ = self.game_.get_state()
        self.frames_ = torch.empty(4, 100, 100, dtype=torch.float)

        self.resize_ = T.Compose([
            T.ToPILImage(),
            T.Grayscale(),
            T.Resize((100, 100)),
            T.ToTensor()
        ])
        self.num_actions_ = self.game_.get_available_buttons_size()


    def reset(self):
        self.game_.new_episode()
        self.state_ = self.game_.get_state()

        img = self.state_.screen_buffer
        img = img.transpose(1, 2, 0)
        img = img[:210, 55:265, :]
        img = self.resize_(img)
        self.frames_ = torch.cat([img, img, img, img], 0).to('cuda')

        return self.frames_.unsqueeze(0)

    def get_state(self):
        return self.frames_.unsqueeze(0)

    def action(self, action):
        a = [0] * self.num_actions_
        a[action] = 1
        reward = self.game_.make_action(a)

        self.state_ = self.game_.get_state()
        if self.state_ is None:
            return 0
        img = self.state_.screen_buffer
        img = img.transpose(1, 2, 0)
        #img = img[:210, 55:265, :]
        img = self.resize_(img).to('cuda')
        self.frames_ = torch.cat([self.frames_, img], 0)
        self.frames_ = self.frames_[:4, :, :]

        return reward

    def done(self):
        return self.game_.is_episode_finished()

    def action_space(self):
        return self.num_actions_
