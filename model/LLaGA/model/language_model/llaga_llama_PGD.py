#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast

from ..llaga_arch import LlagaMetaModel, LlagaMetaForCausalLM
from utils.constants import IGNORE_INDEX


class LlagaConfig(LlamaConfig):
    model_type = "llaga"


class LlagaLlamaModel(LlagaMetaModel, LlamaModel):
    config_class = LlagaConfig

    def __init__(self, config: LlamaConfig):
        super(LlagaLlamaModel, self).__init__(config)


class LlagaLlamaForCausalLM(LlamaForCausalLM, LlagaMetaForCausalLM):
    config_class = LlagaConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlagaLlamaModel(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        graph: Optional[torch.FloatTensor] = None,
        graph_emb: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
        alpha: float = 0.5,
        epsilon: float = 1e-3,
        pgd_steps: int = 5,
        step_size: float = 1e-4,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        if graph_emb is not None:
            graph_emb = graph_emb
            graph_emb.requires_grad = True
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        input_ids_new, attention_mask_new, past_key_values_new, inputs_embeds_new, labels_new = self.prepare_inputs_labels_for_multimodal(input_ids, attention_mask, past_key_values, labels, graph, graph_emb)
        outputs = self.model(
            input_ids=input_ids_new,
            attention_mask=attention_mask_new,
            past_key_values=past_key_values_new,
            inputs_embeds=inputs_embeds_new,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels_new is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels_new[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss(ignore_index=IGNORE_INDEX)
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)
            loss_original = loss_fct(shift_logits, shift_labels)

        # PGD Adversarial Training
        if graph_emb is not None and labels_new is not None:
            graph_emb_adv = graph_emb.clone().detach()
            for _ in range(pgd_steps):
                graph_emb_adv.requires_grad = True
                input_ids_adv, attention_mask_adv, past_key_values_adv, inputs_embeds_adv, labels_adv = self.prepare_inputs_labels_for_multimodal(
                    input_ids, attention_mask, past_key_values, labels, graph, graph_emb_adv
                )
                
                outputs_adv = self.model(
                    input_ids=input_ids_adv,
                    attention_mask=attention_mask_adv,
                    past_key_values=past_key_values_adv,
                    inputs_embeds=inputs_embeds_adv,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )
                
                hidden_states_adv = outputs_adv[0]
                logits_adv = self.lm_head(hidden_states_adv)

                shift_logits_adv = logits_adv[..., :-1, :].contiguous()
                shift_labels_adv = labels_adv[..., 1:].contiguous()

                shift_logits_adv = shift_logits_adv.view(-1, self.config.vocab_size)
                shift_labels_adv = shift_labels_adv.view(-1)
                shift_labels_adv = shift_labels_adv.to(shift_logits_adv.device)
                
                loss_fct = CrossEntropyLoss(ignore_index=IGNORE_INDEX)
                loss_adv = loss_fct(shift_logits_adv, shift_labels_adv)
                with torch.cuda.amp.autocast(enabled=False):
                    loss_adv.backward(retain_graph=True)
                print("graph_emb_adv.grad after backward:", graph_emb_adv.grad)
                perturbation = step_size * graph_emb_adv.grad.sign()
                graph_emb_adv = (graph_emb_adv + perturbation)
                graph_emb_adv = torch.clamp(graph_emb_adv, graph_emb - epsilon, graph_emb + epsilon).detach()
            # print(f"Loss original: {loss_original.item()}, Loss adv: {loss_adv.item()}")
            loss = alpha * loss_original + (1 - alpha) * loss_adv

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "graph": kwargs.get("graph", None),
                "graph_emb": kwargs.get("graph_emb", None),
            }
        )
        return model_inputs

AutoConfig.register("llaga", LlagaConfig)
AutoModelForCausalLM.register(LlagaConfig, LlagaLlamaForCausalLM)