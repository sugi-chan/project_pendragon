import numpy as np
from grab_screen import grab_screen
import cv2
import time
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
from torch.autograd import Variable
import PIL
from siamese_net_model import SiameseNetwork
import torch.nn.functional as F
import pyautogui


from utils import get_card_raw, image_loader, get_predicted_class
from utils import click_location,check_for_chain,brave_chain_checker 
from utils import grab_screen_fgo,get_cards,brave_chain_check,pick_cards_from_card_list
from utils import detect_start_turn


def main():
	turn_counter = 0 
	try:
		while(1):
			time.sleep(3)
			#print('')
			#keypressed = input("Press 1 to continue... q to quit ")
			turn_start = detect_start_turn()
			print('testing_turn',turn_start)			
			
			
			if turn_start == True:

				turn_counter+=1
				#use NPs then check for chains
				#print('move attack')	
				pyautogui.moveTo(1378, 696)
				#print('click attack')
				pyautogui.click() # TEST
				time.sleep(2.0)
				

				if turn_counter > 9:
					time.sleep(1)
					#print('')
					#print('using NPs')
					#check NPs
					#np1 = get_card_raw("NP1", screen)
					#np2 = get_card_raw("NP2", screen)
					#np3 = get_card_raw("NP3", screen)


					click_location("NP1")
					click_location("NP2")
					click_location("NP3")
					#np_cards = [np1,np2,np3]
					#for np in np_cards:
				#		np_chain_list, bool_keep = NP_brave_chain_check(np,card_list,brave_chain_raw_img_list)
				#		if bool_keep == True:#
				#
				#				break


				screen = grab_screen_fgo()
				card_list, brave_chain_raw_img_list = get_cards(screen)

				card_list, brave_chain_raw_img_list = brave_chain_check(card_list,brave_chain_raw_img_list)
				pick_cards_from_card_list(card_list)



			else:
				continue

	except KeyboardInterrupt:
		print('interrupted!')


if __name__ == "__main__":
    main()