# coding=utf-8

# Copyright [2025] [SkywardAI]
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from transformers import GPT2LMHeadModel, pipeline, set_seed

class GPT2Inspector:
    def __init__(self, model_name="gpt2", seed=42):
        set_seed(seed)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.sd_hf = self.model.state_dict()
        self.generator = pipeline('text-generation', model=model_name)

    def print_state_dict_keys(self):
        for k in self.sd_hf.keys():
            print(k, self.sd_hf[k].shape)

    def show_positional_encoding(self, elements=20):
        return self.sd_hf['transformer.wpe.weight'].view(-1)[:elements]

    def generate_text(self, prompt="Hello, I'm a language model,", max_length=30, num_return_sequences=50):
        return self.generator(prompt, max_length=max_length, num_return_sequences=num_return_sequences)

    def check_weight_shapes(self):
        lm_head_shape = self.sd_hf["lm_head.weight"].shape
        wte_weight_shape = self.sd_hf["transformer.wte.weight"].shape
        return lm_head_shape, wte_weight_shape

    def compare_weights(self):
        return (self.sd_hf["lm_head.weight"] == self.sd_hf["transformer.wte.weight"]).all()

    def print_data_ptrs(self):
        print(self.sd_hf["lm_head.weight"].data_ptr())
        print(self.sd_hf["transformer.wte.weight"].data_ptr())

if __name__ == "__main__":
    inspector = GPT2Inspector()
    # inspector.print_state_dict_keys()
    # inspector.print_data_ptrs()
    positional_encoding = inspector.show_positional_encoding()
    generation_results = inspector.generate_text()
    print(generation_results)

    # shape_lm, shape_wte = inspector.check_weight_shapes()
    are_weights_equal = inspector.compare_weights()
    print(are_weights_equal)