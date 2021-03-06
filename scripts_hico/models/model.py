# coding=utf-8
from __future__ import print_function, division
import torch
import torch.nn as nn

import os
import numpy as np
import  sys
sys.path.append("..")
import utils.pool_pairing  as ROI

import torchvision.models as models

lin_size = 1024
ids = 80
context_size = 1024
sp_size = 1024
mul = 3
deep = 512
pool_size = (10, 10)
pool_size_pose = (18, 5, 5)


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size()[0], -1)


class VSGNet(nn.Module):
    def __init__(self):
        super(VSGNet, self).__init__()

        model = models.resnet152(pretrained=True)
        self.flat = Flatten()

        self.Conv_pretrain = nn.Sequential(*list(model.children())[0:7])  ## Resnets,resnext

        ######### Convolutional Blocks for human,objects and the context##############################
        self.Conv_people = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Conv2d(512, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=False),
        )
        self.Conv_objects = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Conv2d(512, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=False),
        )
        self.Conv_context = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Conv2d(512, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=False),
        )

        self.lin_peo = nn.Sequential(
            nn.Linear(1024, 512),

            nn.ReLU(),

        )

        self.lin_obj = nn.Sequential(
            nn.Linear(1024, 512),

            nn.ReLU(),

        )
        self.lin_con = nn.Sequential(
            nn.Linear(1028, 512),

            nn.ReLU(),

        )
        ###############################################################################################

        ##### Attention Feature Model######
        self.conv_sp_map = nn.Sequential(
            # nn.Conv2d(2, 64, kernel_size=(5, 5)),
            nn.Conv2d(3, 64, kernel_size=(5, 5)),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(64, 32, kernel_size=(5, 5)),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.AvgPool2d((13, 13), padding=0, stride=(1, 1)),
            # nn.Linear(32,1024),
            # nn.ReLU()

        )

        self.conv_single_map = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(5, 5)),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(64, 32, kernel_size=(5, 5)),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.AvgPool2d((13, 13), padding=0, stride=(1, 1)),
            # nn.Linear(32,1024),
            # nn.ReLU()

        )

        self.spmap_up = nn.Sequential(
            nn.Linear(32, 512),
            nn.ReLU(),
        )

        self.posemap_up = nn.Sequential(
            nn.Linear(32, 512),
            nn.ReLU(),
        )

        #######################################

        ### Prediction Model for attention features#######
        self.lin_spmap_tail = nn.Sequential(
            nn.Linear(512, 117),

        )

        ##################################################

        ######### Graph Model Basic Structure ########################
        self.peo_to_obj_w = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(),
        )
        self.obj_to_peo_w = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(),
        )

        #################################################################

        # Interaction prediction model for visual features######################
        self.lin_single_head = nn.Sequential(
            # nn.Linear(2048,1),
            # nn.Dropout2d(p=0.5),
            nn.Linear(lin_size * 3 + 4, 1024),
            # nn.Linear(lin_size*3, 1024),
            nn.Linear(1024, 512),

            nn.ReLU(),

        )
        self.lin_single_tail = nn.Sequential(
            # nn.ReLU(),
            nn.Linear(512, 1),
            # nn.Linear(10,1),

        )


        #################################################################

        ########## Prediction model for visual features#################

        self.lin_visual_head = nn.Sequential(
            # nn.Linear(2048, 117),
            # nn.Dropout2d(p=0.5),
            nn.Linear(lin_size * 3 + 4, 1024),
            # nn.Linear(lin_size*3, 1024),
            # nn.Linear(lin_size*3+4+sp_size, 1024),
            nn.Linear(1024, 512),

            nn.ReLU(),
            #  nn.ReLU(),
        )
        self.lin_visual_tail = nn.Sequential(
            nn.Linear(512, 117),

        )

        self.lin_dim_down = nn.Sequential(
            nn.Linear(1536, 1024),
            nn.Linear(1024, 512),
            nn.ReLU(),
        )
        self.w2v_fc = nn.Sequential(
            nn.Linear(80, 512),
            nn.Linear(512, 512),
            nn.ReLU(),
        )
        self.final_obj_to_512 = nn.Sequential(
            nn.Linear(1024, 512),

            nn.ReLU(),

        )
        self.dis = nn.Sequential(
            nn.Linear(17, 512),
            nn.Linear(512, 512),
            nn.ReLU(),
        )
        ################################################

        ####### Prediction model for graph features##################
        self.lin_graph_head = nn.Sequential(
            # nn.Linear(2048, 117),
            # nn.Dropout2d(p=0.5),
            nn.Linear(lin_size * 2, 1024),
            nn.Linear(1024, 512),

            nn.ReLU(),

        )
        self.lin_graph_tail = nn.Sequential(
            nn.Linear(512, 117),

        )

        ########################################

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, pairs_info, pairs_info_augmented, image_id, flag_, phase):
        out1 = self.Conv_pretrain(x)  ###

        rois_people, rois_objects, spatial_locs, union_box, object_one_hot, pose_box = ROI.get_pool_loc(out1, image_id, flag_,    #, distance
                                                                                              size=pool_size,
                                                                                              spatial_scale=25,
                                                                                              batch_size=len(
                                                                                                  pairs_info))
        pose_box = torch.unsqueeze(pose_box, 1)
        ### Defining The Pooling Operations ####### 
        x, y = out1.size()[2], out1.size()[3]
        hum_pool = nn.AvgPool2d(pool_size, padding=0, stride=(1, 1))
        obj_pool = nn.AvgPool2d(pool_size, padding=0, stride=(1, 1))
        context_pool = nn.AvgPool2d((x, y), padding=0, stride=(1, 1))
        #################################################

        ### Human###
        residual_people = rois_people
        res_people = self.Conv_people(rois_people) + residual_people
        res_av_people = hum_pool(res_people)
        out2_people = self.flat(res_av_people)
        ###########

        ##Objects##
        residual_objects = rois_objects
        res_objects = self.Conv_objects(rois_objects) + residual_objects
        res_av_objects = obj_pool(res_objects)
        out2_objects = self.flat(res_av_objects)
        #############    

        #### Context ######
        residual_context = out1
        res_context = self.Conv_context(out1) + residual_context
        res_av_context = context_pool(res_context)
        out2_context = self.flat(res_av_context)
        #################

        ##Attention Features##
        out2_union = self.spmap_up(self.flat(self.conv_sp_map(union_box)))
        ############################

        ##Distance Features## #
        # out2_dis = self.dis(distance)
        ############################

        #### Making Essential Pairing##########
        pairs, people, objects_only, pair_obj_one_out = ROI.pairing(out2_people, out2_objects, out2_context, spatial_locs, pairs_info,
                                                                    object_one_hot)
        ####################################

        ### Muti-stream caculate##########
        single_people = pairs[:, 0:1024]
        single_object = pairs[:, 1024:2048]
        single_context = pairs[:, 2048:]
        single_people = self.lin_peo(single_people)
        single_object = self.lin_obj(single_object)
        single_context = self.lin_con(single_context)  # ??????512
        # pose_box = union_box.split(1, 1)[2]  # [1,1,64,64]
        out2_pose = self.posemap_up(self.flat(self.conv_single_map(pose_box)))
        lin_stream_p = single_people * out2_pose
        lin_stream_o = single_object
        w2v_obj = self.w2v_fc(pair_obj_one_out)
        lin_stream_c = single_context * out2_union
        lin_fix_o = torch.cat((lin_stream_o, w2v_obj), 1)
        lin_final_o = self.final_obj_to_512(lin_fix_o)
        lin_join = torch.cat((lin_stream_p, lin_final_o, lin_stream_c), 1)  # [70,512*4]
        lin_join = self.lin_dim_down(lin_join)

        ###### Interaction Probability##########
        # lin_single_h = self.lin_single_head(pairs)
        # lin_single_t = lin_single_h * out2_union
        lin_single = self.lin_single_tail(lin_join)
        interaction_prob = self.sigmoid(lin_single)
        ####################################################

        ####### Graph Model Base Structure##################
        people_t = people
        objects_only = objects_only
        combine_g = []
        people_f = []
        objects_f = []
        pairs_f = []
        start_p = 0
        start_o = 0
        start_c = 0
        for batch_num, l in enumerate(pairs_info):

            ####Slicing##########
            people_this_batch = people_t[start_p:start_p + int(l[0])]
            no_peo = len(people_this_batch)
            objects_this_batch = objects_only[start_o:start_o + int(l[1])]
            no_objects_this_batch = objects_only[start_o:start_o + int(l[1])][0]
            no_obj = len(objects_this_batch)
            interaction_prob_this_batch = interaction_prob[start_c:start_c + int(l[1]) * int(l[0])]
            if no_obj == 0:

                people_this_batch_r = people_this_batch

                objects_this_batch_r = no_objects_this_batch.view([1, 1024])
            else:
                peo_to_obj_this_batch = torch.stack(
                    [torch.cat((i, j)) for ind_p, i in enumerate(people_this_batch) for ind_o, j in
                     enumerate(objects_this_batch)])
                obj_to_peo_this_batch = torch.stack(
                    [torch.cat((i, j)) for ind_p, i in enumerate(objects_this_batch) for ind_o, j in
                     enumerate(people_this_batch)])
                ###################

                ####### Adjecency###########
                adj_l = []
                adj_po = torch.zeros([no_peo, no_obj]).cuda()
                adj_op = torch.zeros([no_obj, no_peo]).cuda()

                for index_probs, probs in enumerate(interaction_prob_this_batch):
                    # if index_probs % (no_obj+1)!=0:
                    adj_l.append(probs)

                adj_po = torch.cat(adj_l).view(len(adj_l), 1)
                adj_op = adj_po
                # import pdb;pdb.set_trace()
                ##############################

                ###Finding Out Refined Features######

                people_this_batch_r = people_this_batch + torch.mm(adj_po.view([no_peo, no_obj]),
                                                                   self.peo_to_obj_w(objects_this_batch))

                objects_this_batch_r = objects_this_batch + torch.mm(adj_op.view([no_peo, no_obj]).t(),
                                                                     self.obj_to_peo_w(people_this_batch))
            # objects_this_batch_r=torch.cat((no_objects_this_batch.view([1,1024]),objects_this_batch_r))
            #############################

            #### Restructuring ####
            people_f.append(people_this_batch_r)
            people_t_f = people_this_batch_r
            objects_f.append(objects_this_batch_r)
            objects_t_f = objects_this_batch_r

            pairs_f.append(torch.stack(
                [torch.cat((i, j)) for ind_p, i in enumerate(people_t_f) for ind_o, j in enumerate(objects_t_f)]))

            # import pdb;pdb.set_trace()
            ##############################

            ###Loop increment for next batch##
            start_p += int(l[0])
            start_o += int(l[1])
            start_c += int(l[0]) * int(l[1])
        #####################

        people_graph = torch.cat(people_f)
        objects_graph = torch.cat(objects_f)
        pairs_graph = torch.cat(pairs_f)
        ######################################################################################################################################

        #### Prediction from visual features####
        # lin_h = self.lin_visual_head(pairs)
        # lin_t = lin_h * out2_union
        lin_visual = self.lin_visual_tail(lin_join)
        ##############################

        #### Prediction from graph features####

        lin_graph_h = self.lin_graph_head(pairs_graph)
        lin_graph_t = lin_graph_h * out2_union
        lin_graph = self.lin_graph_tail(lin_graph_t)

        ####################################	

        ##### Prediction from attention features #######
        lin_att = self.lin_spmap_tail(out2_union)
        #############################

        return [lin_visual, lin_single, lin_graph, lin_att]  # ,lin_obj_ids]
