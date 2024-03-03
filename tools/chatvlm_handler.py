from transformers import AutoModelForCausalLM
from .base import BaseHandler
from utils.chatglm_utils import *


class ChatVLMHandler(BaseHandler):
    def __init__(self, args):
        super().__init__(args)
        self.vlm_model_path = self.handle_args.get('vlm_model_path')
        self.num_gpus = self.handle_args.get('num_gpus', 2)
        if os.getenv('CUDA_VISIBLE_DEVICES'):
            self.num_gpus = min(self.num_gpus, len(os.environ['CUDA_VISIBLE_DEVICES']))

        self.trust_remote_code = self.handle_args.get('trust_remote_code', True)
        self.device = self.handle_args.get('device', 'cuda:0')

    def init_model(self):
        if self.model is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.vlm_model_path, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(self.vlm_model_path, device_map="cuda", trust_remote_code=self.trust_remote_code)
            self.model = self.model.eval()

    def chat(self, user_input, image, chatbot, history=None, **kwargs):
        self.init_model()
        
        query = self.tokenizer.from_list_format([
            {'image': self.preprocess(image)},
            {'text': user_input},
        ])
        response, history = self.model.chat(self.tokenizer, query=query, history=history)
        chatbot.append((parse_text(user_input), parse_text(response)))
        # image = tokenizer.draw_bbox_on_latest_picture(response, history)
        return chatbot, history

    def chat_stream(self, user_input, image, chatbot, history=None, **kwargs):
        self.init_model()
        chatbot.append((parse_text(user_input), ""))
        query = self.tokenizer.from_list_format([
            {'image': self.preprocess(image)},
            {'text': user_input},
        ])
        for response in self.model.chat_stream(self.tokenizer, query=query, history=history):
            chatbot[-1] = (parse_text(user_input), parse_text(response))
            yield chatbot, history

    def preprocess(self, image):
        return image
        

