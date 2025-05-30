import json
import math
import os

basic_config_path = "./config/config.json"

class Params():
    """
        A customized parameter parser.
    """
    def __init__(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def save(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        return self.__dict__

class AutoConfig():
    def __init__(self, config):
        self.config = config
        init_config = config["init_config"]
        self.pos_norm = init_config["pos_norm"]
        self.pos_bias = init_config["pos_bias"]
        self.v_norm = init_config["v_norm"]
        
        self.rho = config["rho"]
        self.merging_threshold = config["merging_threshold"]
        self.per_step_time = config["per_step_time"]

    def _reset(self, config): 
        self.__init__(config)

    def _config_to_json(self, config, output_path):
        with open(output_path, 'w') as f: 
            f.write(json.dumps(config))

    def _reset_config(self, config, output_path):
        init_config = config["init_config"]
        init_config["pos_norm"] = self.pos_norm
        init_config["pos_bias"] = self.pos_bias
        init_config["v_norm"] = self.v_norm
        config["init_config"] = init_config

        config["rho"] = self.rho
        config["merging_threshold"] = self.merging_threshold
        config["per_step_time"] = self.per_step_time
        self._config_to_json(config, output_path)

    def plan_force(self, scale, output_path):
        self.rho = self.rho * scale
        output_path += f"-force-{scale}.json"
        self._reset_config(self.config, output_path)

    def plan_threshold(self, scale, output_path):
        self.merging_threshold = self.merging_threshold * scale
        output_path += f"-threshold-{scale}.json"
        self._reset_config(self.config, output_path)

    def plan_line_density(self, scale, output_path):
        self.pos_norm = self.pos_norm * scale
        self.pos_bias = self.pos_bias * scale 
        self.merging_threshold = self.merging_threshold * scale
        self.v_norm = self.v_norm/math.sqrt(scale)
        self.per_step_time = self.per_step_time * math.pow(scale, 3/2)
        output_path += f"-line-density-{scale}.json"
        self._reset_config(self.config, output_path)

if __name__ == "__main__":
    basic_config = Params(basic_config_path).dict
    output_path = "./config/auto-config"
    scale = 16
    A = AutoConfig(basic_config)
    A.plan_line_density(scale, output_path)
        

