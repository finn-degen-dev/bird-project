Skip to
content
Kaggle

Create
Search

Home

Competitions

Datasets

Models

Benchmarks

Game Arena

Code

Discussions

Learn

More


View Active Events

Skip to
content
Kaggle

Sign In

Register
1st place solution models (it’s not all BirdNet)

BirdCLEF 2022

Solution Writeup · 1st place · Jun 5, 2022

First of all, thanks to Kaggle Team, @stefankahl and all participants

As you may know, in our final blend we have used BirdNet along with other models provided by me, @ivanpan, @realsleim and @selimsef

In this post I will cover my models and approach

Model Architecture

I was using SED architecture, proposed and used by @tattaka - post

As a backbones I have used:

tf_efficientnet_b3_ns
eca_nfnet_l0
I have changed the first stride from (2,2) to (1,1) in order to have larger (in terms of length and number of frequencies ) output of CNN encoder ( I have taken this trick from @ilu000 SETI)

Model Training

I have trained model on 15 sec chunks and with secondary labels

I have used following augmentations:

GaussianNoise
PinkNoise
OR Mixup on waveforms
BackgroundNoise. For training - this dataset. For finetuning - esc50 + nocall from soundscapes of 2021 BirdClef Comp
Proposed by @selimsef I have used weights (computed by primary_label) for Dataloader and Loss in order to cope with unbalanced dataset (especially for scored_birds)

As for Loss I have used simple BCE on clipwise logits

I was tracking 3 best checkpoints by LB metric and Validation loss and then averaged 3 model weights (kind of naive SWA)

Training stages

For training I have used 2 stage training:

Pretrain on 2021 and 2022 comp data
Finetune on data from pretrain BUT filtered by the next rule - Take samples which contain scored_bird in primary_label OR secondary_labels
Inference

Having SED model I have tried 2 options for inference:

Proposed by @tattaka - using AND rule for long and short clipwise predictions. short prediction - 5 sec, long - 15 sec
Feed model 15 sec chunk BUT apply head only on centered 5 sec reduced CNN image and use max(framewise, dim=time)
Overall second option worked better for me

Choosing threshold. Here I have tried also 2 options:

Use ordinary threshold. Optimal values varied for me from 0.2 - 0.3
Use quantile threshold, originally proposed by @philippsinger post. Optimal value for quantile_tresh was 0.25
tresh = np.quantile(
                        test_model_probs[:, scored_bird_ids].flatten(),
                        1 - quantile_tresh,
                    )
For solo model (5 folds) second worked better and for ensembles first worked better

Validation

I have used 5 CV Stratified training (also maupar sample was splitted on 5 samples in order to have consistent OOF score)
Compute LB metric using next prediction scheme:
For each validation sample - slice it on pieces -> predict each piece -> max(sample_predictions, dim=pieces)
And then compute metric using all_labels=[primary_label] + secondary_labels
Also I have computed this metric only on samples which contain scored_bird in all_labels and taking into account only scored_birds
And finally optimize threshold with step 0.1

Results

tf_efficientnet_b3_ns: Val = 0.87894692; Public = 0.82; Private = 0.78
eca_nfnet_l0: Val = 0.88640918; Public = 0.82; Private = 0.78
Inference Kernel - https://www.kaggle.com/code/ivanpan/fork-of-fork-of-cls-exp-1-870246-021187-967146/notebook?scriptVersionId=96433080
GitHub Repo - https://github.com/Selimonder/birdclef-2022

Project Details
Authors
realsleim
realsleim

Ivan Panshin
ivanpan

Selim Seferbekov
selimsef

Volodymyr
vladimirsydor


Share
4 Comments
Hotness
Takashi Tamura
Posted 4 years ago

393rd in this Competition

Congrats on 1st place! Thanks for sharing your solution. Amazing work!


Reply
Lainey
Posted 3 years ago

210th in this Competition

Thanks a lot, the solution description and code repo!


Reply
Hao
Posted 3 years ago

78th in this Competition

Congratulations for the 1st place, @vladimirsydor. Regarding your 2 stages training, in your 2nd finetune stage training, di d you train all layers of your model, or freeze backbone to avoid overfitting?

Finetune on data from pretrain BUT filtered by the next rule - Take samples which contain scored_bird in primary_label OR secondary_labels


Reply
This comment has been deleted.

Skip to
content
Kaggle

Create
Search

Home

Competitions

Datasets

Models

Benchmarks

Game Arena

Code

Discussions

Learn

More


View Active Events

Skip to
content
Kaggle

Sign In

Register
[Public #1 Private #2] + [Private #7/8 (potential)] solutions. The host wins.

BirdCLEF 2022

Solution Writeup · 2nd place · Jan 25, 2024

As I promised, I will publish my solution right after the competition. This will be a long topic as the two solutions are totally different. As kaggle says > You've been sharing this link too often. We prevent redundant posts to reduce spam. So I put some link in the code block.

Before the solutions, thanks to my teammate @jionie, our second gold medal, and first-time money zone!!!!!!!!!! Cheers!!!!!!!!!!

I will start with our [Private #7/8 (potential)] Solution first since I **promise **you will say the "F" word after seeing the other solution.

[Private #7/8 (potential)] Solution

Training:
Our first solution is based on @kaerunantoka 's public notebook and last year's second-place solution. The training code is pretty the mixture of the two pipelines mentioned above.
Some key points to mention:

data augmentation:
OneOf([ Gain(min_gain_in_db=-15, max_gain_in_db=15, p=0.8), GainTransition(min_gain_in_db=-15, max_gain_in_db=15, p=0.8), ]), OneOf( [ NoiseInjection(p=1, max_noise_level=0.04), GaussianNoise(p=1, min_snr=5, max_snr=20), PinkNoise(p=1, min_snr=5, max_snr=20), AddGaussianNoise(min_amplitude=0.0001, max_amplitude=0.03, p=0.5), AddGaussianSNR(min_snr_in_db=5, max_snr_in_db=15, p=0.5), ], p=0.3, ), AddBackgroundNoise( sounds_path=self.config.BACKGROUND_PATH, min_snr_in_db=0, max_snr_in_db=2, p=0.5 ), Normalize(p=1),

cut mix + mix up

loss function: BCEWithLogits + BCEFocal2WayLoss

hypers setting (n_fft, n_mels, hop_length) refer to our inference notebook.

Our inference kernel and all trained models are [publicly available]
Only one key point to mention: if a bird is detected in the previous or next 5s, we will rank the probs and add the top5 bird species to the current detected birds.

Because this solution was not selected as our final submission, we don't know the actual rank, but as it scores 0.79 in private, it should rank 7-8.

[#1 Private #2] Solution

Actually, I am curious that no one found it, the solution is provided by the host and finally allowed by the host.

If you look at the discussion Meet the host, you will find the host mentioned BirdNet project in the post.
Click the link, and it is easy to find the host's github repo.
Then, find the overlap classes between the host's model and the scored birds, 20 out of 21 scored birds are the same (except aniani).
Modify species_list.txt under example folder accordingly, I have uploaded the modified repo to kaggle here.
Do some post-processing as in our [inference notebook]
Thanks to @ivanpan who opened this issue, the host says we won't enforce the "non-commercial" clause in the license for BirdCLEF, and you can use BirdNET in your submissions. (I don't know if they also benefit from this repo, we can wait for their solution)
Some Notes:
Honestly, we didn't originally intend to use this model as a final submission until we notice the issue mentioned above, and actually, the ensemble of our own model could also reach the gold zone. The answer is provided by the host so that's why I say it is a weird solution. I guess the BirdNet was trained on the data for the public leaderboard but not the private one since there's a huge drop between the public lb to private (0.91 --> 0.84; 0.85 --> 0.78)

The BirdNet is running on CPU less than 2h inference time, while ours cost about 8h to run on GPU. So the host wins, my hope is the first place team doesn't benefit from that model and can beat the host. Looking forward to their solution.

Finally, a big thanks to @stefankahl for hosting this amazing competition and providing the answer as well xD.

Project Details
Authors
Leon
leonshangguan

jionie
jionie


Share
65 Comments
3 appreciation comments
Hotness
Gilles Vandewiele
Posted 4 years ago

11th in this Competition

Congratulations!

I do feel this should have been disclosed by @stefankahl as it seems like the BirdNET model has been trained at least partially on the test data used in this competition.


Reply
Stefan Kahl
Competition Host
Posted 4 years ago

No, it wasn't. Yet, we used other soundscape data and I never realized that this might be an issue for this competition. My guess is, with competition data only, BirdNET would not have scored in the top 10.


Reply
Kramarenko Vladislav
Posted 4 years ago

4th in this Competition

thanks for the story. it is very interesting whether the host used more training records to train the model or not


Reply
Ari
Posted 4 years ago

17th in this Competition

The host used many more. I think it was somewhere around 2434 species in the train set.


Reply
Kramarenko Vladislav
Posted 4 years ago

4th in this Competition

I'm talking about our species. did they use, for example, the test data of this competition for training


Reply
Leon
Topic Author
Posted 4 years ago

2nd in this Competition

Someone asked me why didn't you contact the host already those 20 days ago when you found it or at least 12 days ago when you wrote your comment? -- Because, as I said, we didn't manage to select BirdNet as our final submission so I didn't think that's an issue and did not contact the host. We keep training our own model until the last minute and our own one is ranking 9 which is also in the gold zone. To be honest, we didn't think the BirdNet is allowed until the host said it is allowed. I understand our solution is weird under some gray area, but I don't understand why someone downvotes my comments (and downvotes the 1st place team members comments). This situation happened in Cassava Leaf Disease Classification as well where someone use pre-trained model and reached top places. Kaggle should manage to develop more clear rules and we should keep looking forward to helping avoid this case happen again.


Reply
Ivan Panshin
Posted 4 years ago

1st in this Competition

BirdNet was a huge revelation to me in this competition. I couldn't believe how well it works, so decided to test on last year competition where we got 8th place. Easily managed to get top-1 on private there, but I believe public scores were around silver zone, that's why I wasn't 100% sure it would perform that well on private here.

Glad I was wrong and now we can see that BirdNet gets top-1 on private in both years.

For the sake of honesty, we actually ensembled BirdNet with our models to get some additional boost and spent a couple of months trying to perfect those models (if I'm not mistaken, they score 0.78-0.79 on private). I believe we will be sharing results soon. But anyway, thanks @stefankahl for this competition. I mentioned this last year, I'm gonna mention this again - great orgs! One of the very few on Kaggle who provide great resources here (including notebooks), and simply answer questions to participants without abandoning a competition 1 week in.

Thanks!


Reply
Kramarenko Vladislav
Posted 4 years ago

4th in this Competition

I think this network works better because they had more data. not a few records, as we have.


Reply

43 more replies
Profile picture for anthony
Profile picture for Ivan Panshin
Profile picture for Kramarenko Vladislav
Profile picture for UEMU
+9
kkiller
Posted 4 years ago

89th in this Competition

Despite the fact that the host mistakenly allow the BirdNet model, it's execrable that some Kagglers can find a leak-like situation and secretly exploit it for gold and money prize. This is not a glorious habit and anyone doing it should not be exalted.

Cheers for @maxhalford who could win the RIID 100k worth competition but chooses to timely disclose the leakage.


Reply
S. Tomizawa
Posted 4 years ago

22nd in this Competition

It reminds me of the 2nd place solution of Cassava competition.


Reply
Leon
Topic Author
Posted 4 years ago

2nd in this Competition

Yes exactly,


Reply
anthony
Posted 4 years ago

13th in this Competition

Seems like this competition was useless:( Best results achieved by existed model which also can be used in real world instead of ensembles which we train.


Reply
Stefan Kahl
Competition Host
Posted 4 years ago

No, it was not useless. We will investigate all write-ups, notebooks and working notes to add to our existing model, drawing ideas from everyone's effort. We want to improve bird sound detection and with your participation in this competition, we can do that.


Reply
WANG XIN
Posted 3 years ago

432nd in this Competition

if a bird is detected in the previous or next 5s, we will rank the probs and add the top5 bird species to the current detected birds.

It's really amazing that the information of adjacent time segments can bring up such a boost.


Reply
Tom Denton
Competition Host
Posted 3 years ago

I've also noticed in some recent work that averaging adjacent time windows can be quite helpful. This might in part be due to calls cut off at the window; you may get a misclassification with a partial-call which becomes easier with a small time-shift.

I've also noticed that this averaging tends to improve precision at the expense of recall. If any time-shift creates more ambiguity, then the model's score will decrease with time-shift averaging.


Reply
atfujita
Posted 4 years ago

37th in this Competition

Congrats and thanks for sharing the solution!
I can't always find this kind of information😂


Reply
Kolya Forrat
Posted 4 years ago

8th in this Competition

Congratulations with your score
And I'm happy that you are the only team who found and used it :)


Reply
Kramarenko Vladislav
Posted 4 years ago

4th in this Competition

the first place also found this


Reply
This comment has been deleted.


Appreciation (3)
gilty15
Posted 4 years ago

Thanks for sharing this

jackmtlee
Posted 4 years ago

Thanks for sharing the solution.

Thịnh Lâm
Posted 4 years ago

81st in this Competition

Thanks for sharing and Congrats!

Skip to
content
Kaggle

Create
Search

Home

Competitions

Datasets

Models

Benchmarks

Game Arena

Code

Discussions

Learn

More


View Active Events

Skip to
content
Kaggle

Sign In

Register
3rd place solution

BirdCLEF 2022

Solution Writeup · 3rd place · Jun 16, 2022

First of all, thanks to Kaggle and Cornell Lab of Ornithology for organizing this competition.

I was lucky enough to team-up with UEMU, it was a fruitful team-up with lots of ideas coming from the both sides.
We've worked hard until the end and thanks to that we managed to secure 3rd place, congratz to all of the winners!

Considering the situation, it seems necessary to mention that we didn't use BirdNet model.

External data we used:
freefield1010, aicrowd2020 and nocall part of BirdCLEF 2021 soundscapes to mix 2022 samples with background noise and impove robustness.

Now for our approach, the key points to build strong and reliable pipeline are the following:

Use SED model & training scheme proposed in tattaka's 4th place solution
Use pseudo-labels & hand-labels for small classes in SED model training
Use CNN proposed in 2nd place BirdCLEF 2021 solution
Divide scored birds into two sets and use different loss functions to train models for each one
Augmentations
Let's get down to our best finding during competition

The bird split
When we merged together, we looked on OOFs predictions produced by our models and found out that SED w/ focal-loss performs very differently compared to mentioned CNN trained w/ BCELoss depending on number of training samples

From our observations, SED models w/ focal-loss tend to make more conservative predictions, and due to the loss design they don't miss small classes:

Here in blue you can see SED model w/ focal-loss, and in pink CNN model trained w/ BCE loss
image

The same can be said about advantage of models which used BCE loss during training over models with focal-loss for the large classes
image

Therefore, we divided the birds into two groups according to the number of data and manual inspection of distribution plots of target-data as above, and used different models for them.

It appears that it's optimal to include birds with number of training samples >= 10 to the Group1, and all other birds into Group2.

Group1: 14birds ['jabwar', 'yefcan', 'skylar', 'akiapo', 'apapan', 'barpet', 'elepai', 'iiwi', 'houfin', 'omao', 'warwhe1', 'aniani', 'hawama', 'hawcre'],
for them we ended up using CNN + SED models which were trained using BCE loss.

Group2: 7birds, ['crehon', 'ercfra', 'hawgoo', 'hawhaw', 'hawpet1', 'maupar', 'puaioh'],
for this group we chose to use SED w/ focal-loss.

CNN model training, Group1 birds (slime part)
For the details of architecture of CNN model, please refer to 2nd place BirdCLEF 2021 solution
It's worth to note, that we didn't use temporal mix-up mentioned in the solution.

Since the CNN model was used only for inference on large enough classes, it allowed us to build reliable validation and monitor metrics for Group1 birds only, for these models we used BCE loss to select best models on validation, however with mix-up augmentation model converged on the last epoch, - this fact allowed us to include some models which were trained on full data in the final ensemble.

Training strategy
Use two front-ends:
sr: 32000, window_size: 2048, hop_size: 512, fmin: 0, fmax: 16000, mel_bins: 256, power: 2, top_db=None
sr: 32000, window_size: 1024, hop_size: 320, fmin: 50, fmax: 14000, mel_bins: 64, power: 2, top_db=None
Epochs: 40
backbone: tf_efficnetnet_b0_ns, tf_efficinetnetv2_s_in21k, resnet34, eca_nfnet_l0
Optimizer: Adam, lr=3e-4, wd=0
Scheduler: CosineAnnealing w/o warm-up
Labels: use union of primary and secondary labels
Startify data: by primary label
Augmentations
The ones that definitely helped

mix-up (the most impactful one)
add background noise (same as 2nd place solution 2021)
spec-augment
cut-mix (helped, but just a little)
Didn't work
augment only scored birds
multiply loss for scored bird by 10
use weighted BCE w/ weights proportionally to number of class appearance in dataset
use PCEN
random power as in vlomme's 2021 solution, pitch-shift
coord-conv as in 2nd place rainforest solution
"rating" data didn't introduce much of a difference
SED model training, Group1&2 birds (UEMU part)
The SED model hasn't changed much from the previous 4th solution.
The main difference is the addition of some augmentations.

As a result, we think that the score has improved by 0.06 or more in Public LB.
For the Group1 model and the Group2 model, we changed the loss function, and the other settings were not changed.

We couldn't find a good CV strategy, so most of the settings are decided by watching at public LB.

Training strategy
Use two front-ends:
sr: 32000, window_size: 2048, hop_size: 1024, fmin: 200, fmax: 14000, mel_bins: 224
sr: 32000, window_size: 1024, hop_size: 512, fmin: 50, fmax: 14000, mel_bins: 128
Epochs: 30-40
Cropsize: 10-15s
backbone: seresnext26t_32x4d, resnet34, resnest50, tf_efficientnetv2s
loss_functiuon: BCE2wayloss(Group1), BCEFocal2WayLoss(Group2) as in kaeruru's public note
Optimizer: Adam, lr=1e-3, wd=1e-5
Scheduler: CosineAnnealing w/o warm-up
Labels: primary label=0.9995, secondary label=0.5000, other=0.0025
Startify data: by primary label
Augmentations
GaussianSNR
Nocall Data of trainsoundscape data in the 2021 comp
Spec-augment
Random_CUTMIX as in kaeruru's public note
random power as in vlomme's 2021 solution
Other
How to crop data:
Use Pseudo labeling. We decided the time to crop from the probability distribution estimated by pretrained SED model.
Oversampling for small samples class:
We split the training files by hand and increase data like 5th place solution for some classes('crehon', 'ercfra', 'hawgoo', 'hawhaw', 'hawpet1', 'maupar', 'puaioh',….)
Didn't work
some augmentations(mixup, Randomlowpassfilter, pitch shift, etc)
use PCEN
use weighted BCE w/ weights proportionally to number of class appearance in dataset
use 'rating' data
use 'eBird_Taxonomy_v2021' data
How to choose thresholds:
The default threshold for every of 21 birds was decided by watching LB (in the end we used 0.05 for Group1 birds, but 0.04 gives >0.82 in private LB).

We manually set the threshold for "skylar" bird as 0.35, since our models trained with BCE loss predict it reliably.

The probability distribution of OOFs prediction for non-targetdata when using focal-loss differs depending on the bird (see below figure), so we set the threshold for each bird from Group2 depending on the distribution.
We adopted the value of 91 percentile of the distribution for these birds.

image

Final results
For our best public LB submission we used 8 CNN models, 8 SED models for Group1 birds & 12 SED models for Group2 birds (single model here is either model trained on one fold or on the whole data)

We've also applied time-smoothing post-processing to Group1 birds as in 2nd place BirdCLEF 2021 solution, but it didn't help on private LB, though boosted +0.01~ on public LB

Name	Public LB	Private LB
CNN model (no augs, single fold)	0.7715	0.7278
CNN model (augs, single fold)	0.7761	0.7359
CNN ensemble (CNN only w/ BCE loss)	0.8327	0.7898
SED model (4fold average w/ BCEFocal2wayloss)	0.8339	0.7823
Combine UEMU's SED w/ focal-loss & slime's CNN w/ BCE Loss using bird split mentioned above	0.8532	0.8052
Same as above, but add more CNNs w/ BCE loss and SED w/ BCE loss to group1 birds (best public LB, sub1)	0.8750	0.8126
Safe submission (lower thresholds, sub2)	0.8556	0.8071
Best private LB (add some models which didn't work on public LB)	0.8707	0.8274
We also had around 20 subs which score > 0.82 on private LB and > 0.87 on public LB, but we didn't select them since we chose two subs in the following manner, - one is the best public LB and the other one with much lower thresholds to prevent the shake-up, this sub also happend to be placed 3rd, we're happy :)

Ask your questions! :)

UPD: we further checked the performance of some nets

Name	Public LB	Private LB
ensemble SED only w/ BCE loss (models fine-tuned for Group1)	0.8139	0.7563
ensemble SED only w/ focal-loss	0.8507	0.8135
ensemble CNN only w/ BCE loss (models fine-tuned for Group1)	0.8360	0.7678
The conclusion is that we need some way to determine proper thresholds for predictions in order to see the true perfomance of the models.

We're sure that bird split can bring 0.02-0.03 perfomance boost w/ proper thresholds.

UPD: 06.08.2022
We published the inference kernel: https://www.kaggle.com/code/asaliquid1011/birdclef2022-3rd-place-inference
My part can be found in the following github repository: https://github.com/dazzle-me/birdclef-2022-3rd-place-solution

Project Details
Authors
slime
martynoveduard

UEMU
asaliquid1011


Share
14 Comments
Hotness
KhanhVD
Posted 4 years ago

33rd in this Competition

Congrats @martynoveduard and @asaliquid1011 on 3rd place and becoming master. Nice solution without BirdNet, thanks for sharing!


Reply
Psi
Posted 3 years ago

Congrats on best score without BirdNet and an elegant solution. Happy that our last year's solution was helpful.


Reply
Michael Mortenson
Posted 3 years ago

Where is the best place to learn about SED models?

This is great! Congrats and thanks for sharing.


Reply
UEMU
Posted 3 years ago

3rd in this Competition

Thanks.
This discussion will help you.
https://www.kaggle.com/competitions/rfcx-species-audio-detection/discussion/211007


Reply
Paulo Junqueira
Posted 3 years ago

65th in this Competition

Congratulations! Really creative solution !


Reply
HinePo
Posted 3 years ago

65th in this Competition

Amazing idea spliting the birds by number of samples and using different loss functions. I didn't get how you combined the predictions on the ensemble though. Was it a simple union of the events (bird detections) that crossed each threshold in each model? Congrats @martynoveduard and well deserved prize!


Reply
slime
Topic Author
Posted 3 years ago

3rd in this Competition

Thanks, actually bird split was @asaliquid1011's idea

We trained all our models using 152 classes,
but some models which belong to Group1 made predictions only for birds included in Group1
and other models made predictions for birds from Group2

Then for each group we simply average the probabilities obtained from the models and apply thresholds as described in the solution (0.05 for birds from Group1 except for "skylar", for it we set 0.35, for birds from Group2 thresholds were selected based on OOFs distribution)

Here are raw values obtained for SED w/ focal-loss
{'crehon': 0.0610, 'ercfra': 0.0303, 'hawhaw': 0.0386, 'hawpet1': 0.0491, 'maupar': 0.0724, 'puaioh': 0.0340}


Reply
yokuyama
Posted 3 years ago

5th in this Competition

This is teamwork. Congratulations!


Reply
Tom Denton
Competition Host
Posted 3 years ago

Lovely write-up and some really interesting methods here. Thanks!


Reply
shinmura0
Posted 3 years ago

6th in this Competition

You understands the characteristics of CNN and SED! And a great solution.
Congrats.


Reply
RIYADH HAITHAM
Posted 3 years ago

762nd in this Competition

great work
about the first plot what do you mean by non target is it the prediction as i know the targets just 0 and 1 not proba actully i did not anderstand the plot ?


Reply
UEMU
Posted 3 years ago

3rd in this Competition

This plot show the distribution of the predicted probabilities of the out of fold data when the data splited by 4 folds.
In the graph of hawhaw, 'target ' show the probability distribution of hawhaw for 4 data labeled hawhaw. 'nontarget' show the distribution of hawhaw for about 6000 data labeled other than hawhaw.
The purpose is to observe the trends of TP and FP for each target and model.


Reply
RIYADH HAITHAM
Posted 3 years ago

762nd in this Competition

@asaliquid1011
great
so in the first plot the best is "SED" model with cyan color because it's propa_ distribution "for TP" has low variance and it's mean is large and because it treated all samples the same way correct me if I was wrong
thank you😄


Reply
This comment has been deleted.

Skip to
content
Kaggle

Create
Search

Home

Competitions

Datasets

Models

Benchmarks

Game Arena

Code

Discussions

Learn

More


View Active Events

Skip to
content
Kaggle

Sign In

Register
4th place

BirdCLEF 2022

Solution Writeup · 4th place · May 25, 2022

Thanks to the organizers for another sound competition. My journey to kaggle started with your bird competition and I am becoming a Grandmaster in your bird competition as well. It's sad that a model trained on non-public data won, because of which, it just couldn't be beaten.

I didn't have much time to participate, so my solution is overfitting last year's model 2020
2021

Differences:

First I train on all birds, then I finish training on 21
I select a different lb threshold for each species.
Otherwise, my decision repeats my public decision of previous years
1 fold = 0.78 private lb

Project Details
Author
Kramarenko Vladislav
vlomme


Share
16 Comments
Hotness
Giba
Posted 4 years ago

437th in this Competition

congrats on your solo gold!


Reply
Kramarenko Vladislav
Topic Author
Posted 4 years ago

4th in this Competition

Thank you. This victory was very important, because I became a solo Grandmaster)


Reply
Paritosh Mahto
Posted 3 years ago

Congrats Kramarenko Vladislav!!😊


Reply
nicks
Posted 3 years ago

Great work!


Reply
CPMP
Posted 3 years ago

421st in this Competition

Nice. I was planning to do something similar with my last year models but lacked time/motivation. Well done!

Congrats on this result and your GM title, well deserved!


Reply
osama Hussien
Posted 4 years ago

Great work


Reply
kkiller
Posted 4 years ago

89th in this Competition

Amazing work @vlomme. I kept an eye on your astonishing performances and It's just awesome. Keep it up !


Reply
Allohvk
Posted 4 years ago

32nd in this Competition

First I train on all birds, then I finish training on 21

Thanks for strong solo finish

Do you freeze all layers except last one? Or you created a new model just for training on 21?


Reply
Kramarenko Vladislav
Topic Author
Posted 4 years ago

4th in this Competition

No, I trained all the layers. I tried replacing the head with a 21 class, but it didn't help much.
I also balanced the number of examples in each epoch


Reply

3 more replies
Profile picture for S. Tomizawa
Profile picture for Kramarenko Vladislav
slime
Posted 4 years ago

3rd in this Competition

Have you used pre-trained weights from your 2021 models (trained with 2021 data),
or does the weights come from ImageNet pre training?

If so, did it have any effect or not?


Reply
Kramarenko Vladislav
Topic Author
Posted 4 years ago

4th in this Competition

I didn't use weights since I removed them)
But in the first step, I trained on all birds, which is the same
For me, the pretrain on the ImageNet allowed the model to converge faster, and the pretrain on all birds reduced overfitting because of the small dataset of the Hawaiian birds


Reply
slime
Posted 4 years ago

3rd in this Competition

Make sure to save them, maybe they'll come in handy for the BirdCLEF 2023 competition :)


Reply
Vadim Irtlach
Posted 4 years ago

Кратко и понятно! Уже представляю вас в золотом кольце 😄👍!


Reply
Skip to
content
Kaggle

Create
Search

Home

Competitions

Datasets

Models

Benchmarks

Game Arena

Code

Discussions

Learn

More


View Active Events

Skip to
content
Kaggle

Sign In

Register
5th place solution

BirdCLEF 2022

Solution Writeup · 5th place · May 25, 2022

Many thanks to Kaggle and Cornell Lab of Ornithology for hosting such an interesting competition.

My solution is a reimplementation of the BirdCLEF 2021 2nd place solution.
Special thanks to the new baseline team for publishing the amazing solution. And thanks to @julian3833 for sharing the great baseline!

Model
My final submission is an ensemble of 9 models of different seed/fold/backbone, each model using last year's 2nd place solution method.

The backbones were:

4x eca_nfnet_l0
2x tf_efficientnetv2_s_in21k
2x resnet34
1x convnext_tiny
Oversampling
To increase the number of files for minority classes(N<20), I split the training files by hand.
For example, maupar appears only in one file, but the file contains many different types of its songs and calls.
I cut it into segments of 10-30 seconds using a waveform editor (Audacity) and created multiple training data from a single file.

The cut audio files were further augmented by applying effects such as noise reduction/reverb/gain/etc. to each segment in the waveform editor. Ultimately, I created 5-20 additional sample files per target minority class.

Training
Melspec
window_size=1024, hop_size=320, fmin=50, fmax=14000, power=2, mel_bins=64, top_db=80 (nfnet, effnet, convnext)
window_size=2048, hop_size=512, fmin=16, fmax=16386, power=2, mel_bins=256, top_db=80 (resnet)
Data Augmentation(Waveform)
Background Noise (2020 nocall, 2021 nocall, freefield1010)
GaussianNoise
PinkNoise
NoiseInjection
RandomVolume
TimeShift
Data Augmentation(Image)
SpecAug
CutOut
Lowpass
TranslateY (shift in freq dim, simulated pitch shift)
Optimizer
AdamW, LR=1e-3, weight decay=1e-5, Cosine Anearling with warmup
Loss
BCEWithLogitsLoss
Weighted loss by rating
25-30 epochs
Training target is the union of primary and secondary labels.
All backbones were pretrained on BirdCLEF2021 data.
micro-f1@0.1 for scored_birds was used for validation. Although a bit odd, the correlation between this score and Public/Private LB was not bad.
Post Processing
Probability averaged over previous and next chunks.
Correct thresholds using call/nocall binary classifier probabilities.
Threshold optimization by percentile per clesses.
# thresholds per class
threshold = pd.Series(np.percentile(test_df[SCORED_BIRDS].values, 90, axis=0), index=SCORED_BIRDS)
What Did Not Work
SED models - could not make it past 0.72 for me
CoordConv
MC Dropout
Label Smoothing
Undersampling for majority classes
ViT / Swin backbone
Project Details
Author
yokuyama
yokuyama


Share
9 Comments
Hotness
Psi
Posted 3 years ago

Congrats! Like with 3rd place I am glad that our last year's solution was still working so well here. Thanks for the writeup.


Reply
HinePo
Posted 3 years ago

65th in this Competition

Congrats for this result!

I also tried the convext family but could not train it. I tried resizing the images to 224 x 224 (if I remember it well) in the albumentations step but it still did not solve the size mismatch error in layers. Did you make any pre-process or resizing step to be able to use convnext? I have never used this architecture before. I tried to use it as encoder to subsequent SED model, it did not work…


Reply
yokuyama
Topic Author
Posted 3 years ago

5th in this Competition

Thanks!

This is my implementation, but I don't think I did anything special other than following timm's style.

import timm
import torch

# create convnet
backbone = timm.create_model(
                "convnext_tiny",
                pretrained=True,
                num_classes=0,
                global_pool="",
                in_chans=1,
                drop_path_rate=0.1,
            )
num_features = backbone.num_features

# inference
with torch.inference_mode():
    mel = torch.rand((1,1,64,500))  # B C H W
    y_pred = backbone(mel)  # => (1, 768, 2, 15)

Reply
Julián Peller (dataista0)
Posted 3 years ago

442nd in this Competition

Congratulations! 🎉🎉


Reply
shinmura0
Posted 4 years ago

6th in this Competition

Congrats. And you became competition master!


Reply
yokuyama
Topic Author
Posted 4 years ago

5th in this Competition

Thanks @shinmurashinmura, and congratulations on becoming a Competitions Master too!


Reply
shinmura0
Posted 3 years ago

6th in this Competition

Thank you. We are competition master.


Reply
sht
Posted 4 years ago

Congrats on the solo gold!!


Reply
Giba
Posted 4 years ago

437th in this Competition

Congrats on your solo gold and thanks for sharing your solution.


Reply
Skip to
content
Kaggle

Create
Search

Home

Competitions

Datasets

Models

Benchmarks

Game Arena

Code

Discussions

Learn

More


View Active Events

Skip to
content
Kaggle

Sign In

Register
1st place solution: Correct Data is All You Need

BirdCLEF 2023

Solution Writeup · 1st place · Jun 5, 2023

Hi, Kagglers!

Let's start our journey in the tricky world of Audio Bird Data and Modelling, but before this, a few very important words:

I would like to thank the Armed Forces of Ukraine, Security Service of Ukraine, Defence Intelligence of Ukraine, and the State Emergency Service of Ukraine for providing safety and security to participate in this great competition, complete this work, and help science, technology, and business not to stop but to move forward.

If You Only Knew the Power of A100s GPUs
I have managed to run 294 experiments: half of them with 5 folds and half of them with full data training. So, all in all, many hypotheses were checked and, of course, most of them were rejected :) So let's take a look.

Data, Data is Everywhere
Let's Start from 2023 Training Data
If you take a look at train_metadata["primary_label"].value_counts(), you may notice some strange maximum magic number:

barswa     500
wlwwar     500
thrnig1    500
eaywag1    500
comsan     500
          ... 
lotcor1      1
whctur2      1
whhsaw1      1
afpkin1      1
crefra2      1
Name: primary_label, Length: 264, dtype: int64
Why do we have a maximum of 500 representatives of some species? I do not know the 100% answer, but I have a strong hypothesis - a bug in XC API. I cannot remember the exact place in the code, but the overall problem lies in the data loading pipeline. Here's how it works:

Download meta files - json files.
Iterate over all urls in the meta file(s) and download them.
BUT if you have more than 500 files for one species - on the first stage, you will have several json files (maximum number of files in one json metafile = 500), and here we have the problem! On the second stage, the API takes into account only one json for each species and ignores the next ones, so you will have a maximum of 500 files per species.

NOTE: I am not sure whether it is fixed in the latest version of the API, but I have used a commit from the previous year, and it was there.

From this bug, we can clearly understand that using the fixed API can hugely enrich our training dataset.

Other more boring stuff
2023/2022/2021/2020 competition data
2020 additional competition data
Zenodo
Xeno-Canto
Data Preparation
Training Data
In order to make validation more robust:

Split samples of species with only one representative into 2 splits. This is done in order to have at least one CV split with each species in train AND val splits.
Remove some duplicates manually.
Remove duplicates by the next rule: Two samples have same: duration, author, primary_label.
Additional training data
From the 2023/2022/2021/2020 competition data plus Xeno-Canto data, I have selected only files with this year's primary labels and added them to the final stage of training.

Pretrained Dataset
When I was using only 2023 training data in the final stage of training, pretraining on 2022/2021/2020 competition data boosted the score a lot. But after adding additional training data, pretraining stopped working on the leaderboard (though it still increased local validation). In the last week, I decided to return to pretraining experiments. This granted me one position up in the public leaderboard and two positions up in the private leaderboard - so, Kagglers, don't forget to revisit even rejected hypotheses :)

Why and when did it work? Compared to previous pretraining experiments, I have:

Filtered out 2023 train data duplicates not only by id but also by 'author + primary_label', as was suggested here
Taken species that are present in 2023/2022/2021/2020 competition data + 2020 additional competition data and only if there are more than 10 representatives of the species. Overall, 822 species.
Added additional files for selected species from Xeno Canto.
Zenodo
I have selected nocall regions and used them as background augmentation.

Data Experiments That Did Not Work
Massive pretraining on all Xeno Canto data.
Background noise from 2021 2nd place as Background augmentation
ESC50 as Background augmentation
Selecting only High Quality samples (>=32kHz) from additional data
Maybe some other ideas out of 200+ experiments that I have just forgotten
Validation: Be soft like cmAP, Do not be hard like F1
Finally! We do not have to select a threshold on completely different training data compared to soundscape data, come up with super sophisticated schemes or fall 19 places (as I did in 2021)

I have used pretty much the same validation scheme as in previous years' competitions:

Stratified CV on 5 Folds
Take max prob from each 5 second clip over time across ALL sample
IMPORTANT: For Padded cmAP it is pretty important to take mean across folds, NOT to do Out Of Fold !!!

Of course, absolute numbers of CV and LB are different:

Best Public LB: 0.84444 (4 fours :) )
Best Private LB: 0.76392
Best CV: 0.9083368282233681
But the rank correlation was pretty good. CV improvement in 0.0x (and more) resulted in improvement on LB. I have nearly all CV results for my experiments, so I hope I will have time to publish a paper with a detailed ablation study and a CV-LB correlation study.

Training
I have taken a look at @philippsinger presentation and understood how strongly I was overfitting all the time.

Due to time and device constraints, I have chosen the following scheme:

Validate the hypothesis on CV and submit the first 2-3 folds.
For ensembling retrain on full train data, so you have one model for each setup
Training Details:

50 Epochs
Adam
CosineAnnealing from 1e-4 (or 1e-3) to 1e-6
Focal loss
64 BS
5 second chunk
SUPER IMPORTANT: Class sampling weights
sample_weights = (
    all_primary_labels.value_counts() / 
    all_primary_labels.value_counts().sum()
)  ** (-0.5)
Same setups for pretrain and finetune
Stages:

Pretrain - refer to Pretrained Dataset
Tune only on scored species
Model
Because of computational constraints, we couldn't use the golden rule of Deep Learning: Stack More Layers!

So I have dived a bit in inference optimization techniques:

ONNX - this worked pretty well for me. It improved the inference time slightly and allowed me to reduce the number of custom dependencies in the inference notebook.
Quantization - I spent more than a week experimenting with it, but unfortunately, I had no success :(
openvino - I didn't use or try this, I just read about it the 2nd place description and burnt my chair
Overall, my final submission is an ensemble of 3 Sound Event Detection (SED) models with the following backbones:

eca_nfnet_l0 (2 stages training; Start LR 1e-3)
convnext_small_fb_in22k_ft_in1k_384 (2 stages training; Start LR 1e-4)
convnextv2_tiny_fcmae_ft_in22k_in1k_384 (1 stage training; Start LR 1e-4)
It was pretty important to tweak the starting learning rate for different architectures!!!

Augmentations
I was pretty picky about augmentation selection, so my final models used next ones:

Mixup : Simply OR Mixup with Prob = 0.5
BackgroundNoise with Zenodo nocall
RandomFiltering - a custom augmentation: in simple terms, it's a simplified random Equalizer
Spec Aug:
Freq:
Max length: 10
Max lines: 3
Probability: 0.3
Time:
Max length: 20
Max lines: 3
Probability: 0.3
Small inference tricks
Using temperature mean: pred = (pred**2).mean(axis=0) ** 0.5
Using Attention SED probs * 0.75 + Max Timewise probs * 0.25
All these gave marginal improvements but it is was a matter of first 3 places :)

Other stuff that created a carbon footprint but did not improve my LB score
This section will be far from complete but let's add something that I have in mind now:

2021 2nd place model. I have tried (like I did in 2022) but unfortunately it did not work for me
Pretrain on whole Xeno Canto
Train on larger chunks. The same result occurred if I inferred on smaller chunks or on same length chunks
Colored Noise augmentations
CQT or LEAF
Specific finetuning: smaller LR, smaller number of epochs, freeze backbone, different LRs for backbone and head
Loss on Attention SED probs + Loss on Max Timewise probs
Deep Supervision
Different alpha for MixUp
Transformer architectures. For example ECAPA TDNN
Closing words
I hope you have not fallen asleep while reading. Finally, I want to thank the entire Kaggle community, congratulate all participants and winners.
Special thanks to Cornell Lab of Ornithology, LifeCLEF, Google Research, Xeno-canto, @stefankahl, @tomdenton, @holgerklinck. All of you were super active in discussions, shared datasets and interesting materials, answered all questions, and of course, prepared such a cool competition!"

Resources
Inference Kernel : https://www.kaggle.com/code/vladimirsydor/bird-clef-2023-inference-v1/notebook
GitHub : https://github.com/VSydorskyy/BirdCLEF_2023_1st_place
Paper : TBD

Project Details
Author
Volodymyr
vladimirsydor


Share
9
23 Comments
1 appreciation comment
Hotness
Yassine Alouini
Posted 2 years ago

Awesome write-up! Curious if the paper has been published?


Reply
4
Mohamed Eltayeb
Posted 3 years ago

99th in this Competition

Congratulations! Very deserved winning.
Just one question, could you elaborate more on this:
"IMPORTANT: For Padded cmAP it is pretty important to take mean across folds, NOT to do Out Of Fold !!!"
Thanks.


Reply
Volodymyr
Topic Author
Posted 3 years ago

1st in this Competition

If you compute Out Of Fold and compute metric ones - you will have only 5 100% correct rows for each class BUT if you compute metric on each fold separately you will have 5 100% correct rows for each fold
So in first case you will have lower metric comparing to the second one


Reply
Hakan Dogan
Posted 3 years ago

669th in this Competition

Many congratulations and thanks a lot for sharing your approach. Would be grateful if could elaborate on where the "sample_weights" come in play. Do you use it to balance the data per class at each epoch, so that not using all rows of the train_df?


Reply
Volodymyr
Topic Author
Posted 3 years ago

1st in this Competition

Yep, not all data training samples is used in each epoch

If we are talking technically, how to do it - it is pretty simple:

compute sample_weights, as I showed and you will have primary_label:weight mapping
apply this mapping on each train sample and you will have List of weights - weights_list
and finally create torch sampler: torch.utils.data.WeightedRandomSampler(weights_list, len(weights_list))

Reply
Aphysict
Posted 3 years ago

12th in this Competition

Congratulations and thanks for sharing! "pretraining stopped working on the leaderboard (though it still increased local validation)", this happens to me too. However, pretraining doesn't improve my CNN model on private LB. I guess SED is more powerful when you have enough data.


Reply
Teemo
Posted 2 years ago

35th in this Competition

Congratulations!
Can you explain a little about how to adjust learning rate, choose scheduler, loss, and epochs?
Thx a lot!


Reply
Nikita Komarov
Posted 2 years ago

1118th in this Competition

Congratulations!


Reply
Fernando Sckaff
Posted 2 years ago

Congratulations, and thank you for sharing your code!


Reply
Pardeep Singh
Posted 2 years ago

447th in this Competition

Congrats on Winning the competition yet again @vladimirsydor and thanks for the detailed writeup. Looking forward to notebooks! 🎉🙌🙏


Reply
Amir Bralin
Posted 2 years ago

It was a good read. And congratulations!


Reply
S.Lalitha Bhavani
Posted 2 years ago

Congratulations! and thank you for sharing your insights


Reply
IamnotPi
Posted 2 years ago

Congratulations!


Reply
Greg McGlaun
Posted 2 years ago

Very insightful!


Reply
Suraj
Posted 2 years ago

581st in this Competition

Kudos @vladimirsydor for winning the competition! Very informative description of the winning approach, too !


Reply
yukiya
Posted 2 years ago

70th in this Competition

Congratulations and thank you for sharing
Could you share how you convert your model to onnx ? I can convert my tensorflow model, but torch model keeps spitting error.
Slava Ukraini !


Reply
Nooruddin Hyderabadwalla, CFA
Posted 3 years ago

Congratulations, and thank you for sharing your approach @vladimirsydor - I feel like whilst I'm learning this, I'm only able to grasp your solution partially. It still feels like a big opportunity to be in the company of people with so much knowledge to address these difficulties, and so much humility to share their findings as well.


Reply
Aaditya Bansal
Posted 3 years ago

284th in this Competition

Congratulations on winning ✨ and Thank you for sharing your approach, I am just a beginner in Deep Learning, do you have any tips as to how I can approach complex problems like this.


Reply
Matt OP
Posted 3 years ago

It's cool to see that the 1st place solution took advantage of removing the duplicates from the training data to make the validation more robust. Congratulations @vladimirsydor!


Reply
atfujita
Posted 3 years ago

4th in this Competition

Congratulation! Thank you for teaching us your extensive experiments and deep insights.


Reply
JEANMPIA
Posted 3 years ago

88th in this Competition

Very impressive solution, I certainly didn't even think about a bug in the API that limitted to 500 samples ! I'm impressed by the curiosity of the winners every comp.


Reply
Harshit Sheoran
Posted 3 years ago

9th in this Competition

Congratulations on very much deserved solo victory


Reply

Appreciation (1)
KAVINKUMAR_VS_2004
Posted 2 years ago

Thanks for the great tutorial!!

Skip to
content
Kaggle

Create
Search

Home

Competitions

Datasets

Models

Benchmarks

Game Arena

Code

Discussions

Learn

More


View Active Events

Skip to
content
Kaggle

Sign In

Register
2nd place solution: SED + CNN with 7 models ensemble

BirdCLEF 2023

Solution Writeup · 2nd place · Apr 4, 2024

Congratulations to all the winners! Thanks to Kaggle and Cornell Lab of Ornithology for hosting this interesting competition.

This is my first solo gold medal and I am glad to have this result.

This competition shared a lot of similarity to the past BirdClef competitions(2020/2021/2022). Thus I spent a lot of time gathering the solution shared by the top teams in the past competitions. Special thanks to all of you for sharing such important information!

Let me briefly introduce my solution. I will update the solution for more details in a couple of days.

Most important (7 models ensemble!)
Please see the notebook below.
openvino is all you need!!

Training data
Here is my training data.

2023/2022/2021/2020 competition data
2020 additional competition data
additional training data from xeno-canto, including 2023 comp species in both foreground and background(records with 2023 comp species only in background which is less than 60 seconds are included).
I intended to collect more records from ebird site, but I realized that ebird data is not public and cannot be used. I asked the host and confirmed that. Thanks @tomdenton answering my questions.

Thus, my training pipeline does not contain records from ebird site.

Model Architecture
First, I used SED architecture. The same as yours.

backbones are:

tf_efficientnetv2_s_in21k
seresnext26t_32x4d
tf_efficientnet_b3_ns
All of them are trained on 10sec clip.

Second, I used CNN proposed by 2nd place of 2021 competition

backbones are:

tf_efficientnetv2_s_in21k
resnet34d
tf_efficientnet_b3_ns
tf_efficientnet_b0_ns
All except b0 are trained on 15sec clip. b0 is trained on 20sec clip.

Pseudo Labeling and Hand Labeling
I have used SED model to generate pseudo label and extracted the potential nocall using quantile threshold. Then, I hand labeled the potential nocall by hearing the record. I hand labeled about 1800 records but did not see improvement. Maybe the pseudo label contains more FP rather than FN. I did not have time to further investigate the prediction.

Model Training
augmentations:

GaussianNoise
PinkNoise
Gain
NoiseInjection
Background Noise(nocall in 2020, 2021 comp + rainforest + environment sound + nocall in freefield1010, warblrb, birdvox)
PitchShift
TimeShift
FrequencyMasking
TimeMasking
OR Mixup on waveforms
Mixup on spectrograms.
With a probability of 0.5 lowered the upper frequencies
self mixup for records with 2023 species only in background.(60sec waveform -> split to 6 * 10sec -> np.sum(audios,axis=0) to get a 10sec clip)
I have used weights (computed by primary_label and secondary_labels) for Dataloader in order to cope with unbalanced dataset.

Training stages
For training I have used 2 stage training:

Pretrain on all data(834 species).
Finetune on 2023 species(264 species).
In both stages, I first train model with CrossEntropyLoss, and then train on with BCEWithLogitsLoss(reduction='sum'). Model converges faster with CrossEntropyLoss than BCEWithLogitsLoss, but BCEWithLogitsLoss gives better score.

To give more diversity, models are trained on different windows and different mixup rate, and some of them only trained on CrossEntropyLoss. And also 3 of the models are fintuned on 30s clip.

CV strategy
For each validation sample - slice the first 60 seconds to pieces -> predict each piece -> max(sample_predictions, dim=pieces).
CV does not show correlation with LB, but it seems that the right ways to improve the LB are those which do not significantly decrease CV. So I monitored the CV when tuning the pipeline.

Inference
For SED model, feed model 10 sec chunk BUT apply head only on centered 5 sec reduced CNN image and use max(framewise, dim=time).

Also, tta(2s) is used for SED model.

Important: convert pytorch model to openvino model significantly reduce inference time(about 40%). (eca_nfnet_l0 backbone ONXX cannot be converted to openvino because the stdconv layer in timm use train mode of F.batch_norm in forward method). That is the magic of ensembling 7 models.

Ensemble
I spent quite a lot of time understanding the metrics. The ensemble is as followed:

(weighted average, 0.84 on LB, 0.76 on private) Apply weighted average on raw logit. This ensemble does not make sense for me because the output logit of models differ and should not be simply added, otherwise the result is biased. But considering that the absolute value of logit may also contribute to the score and it does give the best LB, so I choose it for final submission to have a gamble.
(rank average, 0.83 on LB, 0.75 on private) Convert the logit to rank and apply weighted average on rank. I think this is the reasonable way to ensemble, considering that I don't have a reliable CV. Be careful that in this comp submission is padded with 5 rows of 1, thus the ranking should start from 0 to prevent the largest ranking to be 1 after convert rankings to percentile form.(For example, 0, 0.333, 0.666 is good while 0.333, 0.666, 1.0 is bad. otherwise the score will decrease due to the potential false positive of those ranked 1.0)
Basically all of my single model's performance reaches about 0.81-0.82 on LB, so the weight of each model is similar to give diversity although higher public lb requires large weight of sed v2s and sed b3_ns.

Looking at the private LB, weighted average seems to be better. emmm……Why?

What did not work
contrast
EMA
BirdNet
adjusting logit according to previous and next 5s clip
q transform
Inference Notebook: https://www.kaggle.com/code/honglihang/2nd-place-solution-inference-kernel
github: https://github.com/LIHANG-HONG/birdclef2023-2nd-place-solution

Project Details
Author
RihanPiggy
honglihang


Share
2
35 Comments
Hotness
Jonathan Carlson2
Posted 2 years ago

Hello!
Thanks for the very nice written code!

When i download the .csv file, and rename it to train.csv and put it in ./input, i get this error when trying to train:
KeyError: 'duration'
It comes from
modules/preprocess.py", line 104, in preprocess
(df["duration"] <= cfg.background_duration_thre)
Any idea on how to fix this?


Reply
HungryLearner
Posted 3 years ago

52nd in this Competition

Congratulations…..

Ensembling trick of openvino is indeed something to add to my knowledge arsenal. Great work!!!


Reply
RihanPiggy
Topic Author
Posted 2 years ago

2nd in this Competition

Thank you! It is also my first time to try openvino.


Reply
Gaurav Srivastava
Posted 3 years ago

233rd in this Competition

Amazing 💯 Congratulations and thanks for sharing 😊


Reply
RihanPiggy
Topic Author
Posted 2 years ago

2nd in this Competition

Thanks! It is really a long journey for me and I slept for two days to recover from lack of sleep


Reply
2
Ayça Güler
Posted 6 months ago

Thanks for your contrubition. I have a question about re-producing the pipeline. Can I produce the same results via github or via kaggle notebook. As I understood, I need to construct the directory in github on my own hardware or maybe on colab. But, I can get the same result from the kaggle notebook by running it on kaggle kernel. @honglihang


Reply
Amee Assad
Posted 2 years ago

Here a little late, but…
Your solution is incredibly inventive and probably the easiest to follow compared to all the ones I've viewed from the last 3 years. I hope to use it to finetune on just 2 bird species. I was wondering if you still have your .ckpt files that i could use, so that I don't have to train all over again? Thanks :)


Reply
Pardeep Singh
Posted 2 years ago

447th in this Competition

Great stuff @honglihang congrats on your first solo gold and thanks for sharing this writeup + code 🎉 🔥


Reply
Joao Pedro Rebeschini
Posted 2 years ago

Awesome job!!


Reply
Saleh Ahmad
Posted 2 years ago

That is one amazing solution to the competition!


Reply
Suraj
Posted 2 years ago

581st in this Competition

Awesome approach to use openvino for boosting inference times over ensembling @honglihang ! Congrats !


Reply
Muhammad Bilal Hussain
Posted 3 years ago

@honglihang I wanted to reach out and express my heartfelt gratitude for sharing your incredible 2nd place solution in the recent competition. Your approach, combining Sound Event Detection (SED) with Convolutional Neural Networks (CNN) and utilizing an ensemble of seven models, is truly remarkable.


Reply
RihanPiggy
Topic Author
Posted 2 years ago

2nd in this Competition

Thanks you! We should all thanks to those who shared these amazing stuffs in the past few years.


Reply
Jan Schlüter
Posted 3 years ago

72nd in this Competition

Very nice, congratulations and thanks for the quick writeup! Will look into openvino as well, nice find.

I hand labeled about 1800 records but did not see improvement.

So in the end, you neither used manual nor pseudo-labels? I experimented with pseudo-labels on the test set (with much weaker models than the top solutions), but hardly got an improvement either, hence I'm curious.

self mixup for records with 2023 species only in background

Interesting idea! What was the intuition behind that? Make it more likely that the bird in question is actually part of the 10s clip? How much did it help, compared to other tricks?

In both stages, I first train model with CrossEntropyLoss, and then train on with BCEWithLogitsLoss(reduction='sum').

Again, interesting idea, will keep this in my head!

For SED model, feed model 10 sec chunk BUT apply head only on centered 5 sec reduced CNN image and use max(framewise, dim=time).

This sounds expensive. Does this double the inference time or did you feed longer chunks than 10s and then extracted multiple 5sec chunks from that?

Also, tta(2s) is used for SED model.

tta = test-time augmentation? Was there even time for that? :-O

Again, congratulations for your work!


Reply
RihanPiggy
Topic Author
Posted 2 years ago

2nd in this Competition

Hi, thanks for your question.

I hand labeled about 1800 records but did not see improvement.

I used both pseudo label and hand label, although there is no improvement seen in public LB, I have not confirm whether it has improvement in private LB.

self mixup for records with 2023 species only in background.

The idea comes from that I found mixup in waveform is effective. So I think I can take use of the records with 2023 species only in background by applying the same idea because I no longer need to care about where the bird call appears in background by doing this mixup to fully use the record. I have not confirmed whether it is useful, but there was a marginal improvement in the score.

For SED model, feed model 10 sec chunk BUT apply head only on centered 5 sec reduced CNN image and use max(framewise, dim=time).

Yes, it is expensive which double the inference time. So I only use 10s, not any longer…

Also, tta(2s) is used for SED model.

TTA is only applied on attention head so it is super fast. The encoder extract the 10s feature and the target 5s is in the center of the matrix (2.5s to 7.5s). For tta, I take the feature from 1.5s to 6.5s and 3.5s to 8.5s.


Reply
muhinyia
Posted 3 years ago

511th in this Competition

Congratulations!

Amazing stuff!


Reply
RihanPiggy
Topic Author
Posted 2 years ago

2nd in this Competition

Thank you!


Reply
MICADEE
Posted 3 years ago

328th in this Competition

Incredible !!! Congratulations 🎉.


Reply
RihanPiggy
Topic Author
Posted 2 years ago

2nd in this Competition

Thanks! Happy kaggling!


Reply
atfujita
Posted 3 years ago

4th in this Competition

Congratulations and thanks for sharing! I would like to learn Openvino from your notebook.


Reply
RihanPiggy
Topic Author
Posted 2 years ago

2nd in this Competition

Thank you! The same to you. Your solution is super incredible and intersting. I will take a deep look into your solution after I get enougth sleep~

By the way, have you already got the winner email from the kaggle team? I haven't…


Reply
atfujita
Posted 2 years ago

4th in this Competition

Thanks!
I also haven't got it yet.


Reply
RihanPiggy
Topic Author
Posted 2 years ago

2nd in this Competition

Thanks for your reply.
I still haven't received yet, maybe it cost sometime….


Reply
Aaditya Bansal
Posted 3 years ago

284th in this Competition

Congratulations ✨ and thanks for sharing, I am just a beginner getting started with Deep Learning this year only, for me it is both fascinating and intimidating how you can figure out so many things while approaching a problem. Do you have any tips about how I can become more efficient in problem solving like this. Thank you for your time.


Reply
RihanPiggy
Topic Author
Posted 2 years ago

2nd in this Competition

Thank you! I started kaggle about 2 years ago and also had a hard time to understand the solutions. I think it is a process of try and error based on the knowledge shared by others. It is better to get new ideas through reading about the papers and the past top solutions.


Reply
Mohammad Albaddawi
Posted 3 years ago

381st in this Competition

Congratulations and thanks for sharing


Reply
RihanPiggy
Topic Author
Posted 2 years ago

2nd in this Competition

Thank you! happy kaggle


Reply
sqrt4kaido
Posted 3 years ago

49th in this Competition

Congratulations on achieving second place as a solo participant! Incorporating "heavy" augmentations like pitch shifting seems like it would significantly extend the training time. Have you taken any measures to address this?


Reply
RihanPiggy
Topic Author
Posted 2 years ago

2nd in this Competition

Thank you. I appreciate you sharing the data in 2020 comp and I took full use of it. I used torch-audiomentations to perform pitch shift and time shift on GPU


Reply
Maximiliano Diaz Battan
Posted 3 years ago

75th in this Competition

Congratulations, and thanks for sharing! We tried onnx but not openvino. Onnx worked very well individually, but when we tried to ensemble the models, the results were very bad, almost all the time we ended up with a timeout or some kind of error, probably because we made some mistake during the ensemble. For that reason, we skip onnx, and our ensemble was with only 4 Pytorch models.


Reply
RihanPiggy
Topic Author
Posted 2 years ago

2nd in this Competition

Also congratulations to you! Thanks for sharing the information about onnx. I am also new to onnx and openvino and maybe it is lucky for me to try openvino at first because I learnt about openvino in a project at work


Reply
liuzhangzhen
Posted 3 years ago

31st in this Competition

Congratulations, thanks for sharing


Reply
RihanPiggy
Topic Author
Posted 2 years ago

2nd in this Competition

Thank you and congratulations to you too!


Reply
Harshit Sheoran
Posted 3 years ago

9th in this Competition

Congratulations! It must have been a fulfilling journey :)


Reply
RihanPiggy
Topic Author
Posted 2 years ago

2nd in this Competition

The same to your team! Congratulations! It has been a really a fulfilling(and short sleep) journey for me since the RSNA.
Looks that you have made birdnet work, I will read your solution in detail after getting enough sleep.


Reply
Sinan Calisir
Posted 3 years ago

55th in this Competition

Huge congrats!!! 🎉🎉 Thanks for sharing, I was really curious about the top solutions. I'll read this in detail after having my (missing) night sleep.


Reply
RihanPiggy
Topic Author
Posted 2 years ago

2nd in this Competition

Thank you! Also congrats to you!
The other top solutions are very incredible to me. I will also read in detial after getting back to sleep..


Reply
Skip to
content
Kaggle

Create
Search

Home

Competitions

Datasets

Models

Benchmarks

Game Arena

Code

Discussions

Learn

More


View Active Events

Skip to
content
Kaggle

Sign In

Register
3rd place solution: SED with attention on Mel frequency bands

BirdCLEF 2023

Solution Writeup · 3rd place · Jul 8, 2025

First of all, thanks to the organizers Sohier Dane, Stefan Kahl, Tom Denton, Holger Klinck and all involved institutions (Kaggle, Chemnitz University of Technology, Google Research, K. Lisa Yang Center for Conservation Bioacoustics at the Cornell Lab of Ornithology, LifeCLEF, NATURAL STATE, OekoFor GbR and Xeno-canto).

This year’s competition was a welcome change compared to previous challenges. The cMAP evaluation metric eliminated the need for threshold tuning, while the inference time limit encouraged a focus on efficient models with a good balance between accuracy and speed.

In this post I want to briefly introduce some aspects of my solution. A more detailed description will be provided later as an update or in the upcoming working note.

Quick summary
Modified SED architecture with attention on frequency bands
Addressing domain shift with reverb augmentation
Using freezed TorchScipt models and precalculated inputs to speed up inference
Addressing fluctuating inference time by setting a timer in inference notebook
Datasets
2021/2023 competition data
2020 extended xeno-canto data
2023 extended xeno-canto data (all files with 2023 species as primary label)
BirdCLEF 2019 soundscapes (2021 species only & nocall/noise)
DCASE 2018 Bird Audio Detection Task (nocall/noise)
Some nocall/noise files from datasets of previous competitions/solutions
Data preparation
Convert files to 32 kHz (if necessary)
Convert extended (downloaded) xeno-canto files to FLAC
Add duration information for each file to dataset metadata
Add first 10 seconds interval of all xeno-canto files to training set
Split training set into 8 folds (but mostly only 3 folds were used)
Model input
Log Mel spectrogram of 5 second audio chunks (n_fft= 2048, hop_length=512, n_mels=128, fmin=40, fmax=15000, power=2.0, top_db=100)
Normalized to 0…255
Converted to 3 channel RGB image
Model backbone/encoder architectures (from timm)
tf_efficientnet_b0_ns
tf_efficientnetv2_s_in21k
I also tried resnet50, resnet152, tf_efficientnet_b2_ns, tf_efficientnet_b3_ns, tf_efficientnet_b4_ns, efficientformer_l3, tf_efficientnetv2_m_in21k, densenet121 and eca_nfnet_l0 but none of those were included in inference ensemble because in my case tradeoff between performance and inference time was not as good as for EffNetB0 or EffNetV2s.

All models used pretrained ImageNet weights and served as feature extractor combined with a custom classification head. As classifier I used a modified SED head with attention on frequency bands instead of time frames. The intuition behind this is, that species in soundscapes often occupy different frequency bands. In original SED architecture, feature maps representing frequency bands are aggregated via mean pooling and attention is applied on features representing time frames. If attention is instead applied on frequency bands it can help to distinguish species vocalizing at the same time but with different pitch. The modification can be achieved simply by rotating the Mel spectrogram by 90 degrees before feeding it to the original SED network.

Data augmentation (esp. to deal with weak/noisy labels and domain shift between train/test set)
Select 5s audio chunk at random position within file:
Without any weighting
Weighted by signal energy (RMS)
Weighted by primary class probability (using info from pseudo labeling)
Add hard/soft pseudo labels of up to 8 bird species ranked by probability in selected chunk
Random cyclic shift
Filter with random transfer function
Mixup in time domain via adding chunks of same species, random species and nocall/noise
Random gain of signal amplitude of chunks before mix
Random gain of mix
Pitch shift and time stretch (local & global in time and frequency domain)
Gaussian/pink/brown noise
Short noise bursts
Reverb (see below)
Different interpolation filters for spectrogram resizing
Color jitter (brightness, contrast, saturation, hue)
In soundscapes, birds are often recorded from far away, resulting in weaker sounds with more reverb and attenuated high frequencies (compared to most Xeno-canto files where sounds are usually much cleaner because the microphone is targeted directly at the bird). To account for this difference between training and test data (domain shift), I added reverb to the training files using impulse responses, recorded from the Valhalla Vintage Verb audio effect plugin. During training, I randomly selected impulse responses and convolved them with the audio signal with a 20% chance, using a dry/wet mix control ranging from 0.2 (almost dry signal) to 1.0 (only reverb).

I didn’t use pretraining followed by finetuning, instead I trained on all 2021 & 2023 species + nocall (659 classes). Background species were included with target value 0.3. For inference, predictions were filtered to the 2023 species (264 classes).

Speed up inference and deal with submission time limit
Due to variations in hardware and CPUs used to run inference notebooks, the number of models that could be ensembled varied. To prevent submission timeouts, I set a timer in the notebook to ensure completion within the 2-hour limit. If the timer reached approximately 118 minutes, inferencing was stopped and results were collected for models and file parts predicted up to that point. Results for unfinished models/file parts were masked before averaging predictions. Using this method, I couldn’t determine the exact number of models that could be ensembled. In early submissions, I could only ensemble 3 models without risking timeouts. Later, I prioritized inference speed over model diversity by using models with the same input (no variation in FFT size, number of Mel bands etc.). Now I could precalculate and save Mel spectrogram images to RAM for all test files in advance and reuse those for all models. I also converted models to TorchScript. With these optimizations, I could ensemble at least 7 models, depending on architecture (e.g. 4x EfficientNetB0 + 3x EfficientNetV2s) without setting a timer.

My best single model used an EfficientNetV2s and scored 0.83386 on public leaderboard (0.74104 on private LB). The best single model with highest score on private leaderboard used a ResNet50 backbone (0.7482 private LB / 0.83288 public LB). My best ensemble on private LB (0.76365) was a mix of 8 models (5x EfficientNetB0 + 3x EfficientNetV2s) with simple mean averaging of single model predictions.

Some things I tried but gave up on because I couldn’t get them to work well enough
Model soup
MultiLabelSoftMarginLoss (instead of BCEWithLogitsLoss)
Knowledge Distillation
Finetuning using only 2023 species data
Converting models to ONNX or openvino format (speed up was only achieved for small batch sizes)
Any postprocessing (e.g. amplify probabilities of detected species in neighboring windows or entire file)
Citations
Kong, Qiuqiang, Yin Cao, Turab Iqbal, Yuxuan Wang, Wenwu Wang, and Mark D. Plumbley. "PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition." arXiv preprint arXiv:1912.10211 (2019).
Code for PANNs paper
S. Adavanne, H. Fayek & V. Tourbabin, "Sound Event Classification and Detection with Weakly Labeled Data", Proceedings of the Detection and Classification of Acoustic Scenes and Events 2019 Workshop (DCASE2019), pages 15–19, New York University, NY, USA, Oct. 2019
Introduction to Sound Event Detection by Hidehisa Arai
Lasseck M (2019) Bird Species Identification in Soundscapes. In: CEUR Workshop Proceedings.
https://xeno-canto.org/
timm (PyTorch Image Models)
Audiomentations
Inference Notbook
https://www.kaggle.com/mariotsaberlin/bc23-3rd-place-solution-refactored

Working Note
Lasseck M (2023) Bird Species Recognition using Convolutional Neural Networks with Attention on Frequency Bands. In: CEUR Workshop Proceedings.

Project Details
Author
adsr
mariotsaberlin


Share
4 Comments
Hotness
yukiya
Posted 2 years ago

70th in this Competition

The modification can be achieved simply by rotating the Mel spectrogram by 90 degrees before feeding it to the original SED network.

I am in love with this trick. Thank you very much for sharing!


Reply
Pardeep Singh
Posted 2 years ago

447th in this Competition

Congrats on getting solo gold in this competition @mariotsaberlin Great work, thanks for sharing a detailed writeup 🙏


Reply
sean chenXinyu
Posted 7 months ago

So why don't you put your pretrained-model checkpoint.pt under input files? I cannot reproduce it at all


Reply
sean chenXinyu
Posted 7 months ago

very helpful for me to complete my github


Reply
Skip to
content
Kaggle

Create
Search

Home

Competitions

Datasets

Models

Benchmarks

Game Arena

Code

Discussions

Learn

More


View Active Events

Skip to
content
Kaggle

Sign In

Register
4th Place Solution: Knowledge Distillation Is All You Need

BirdCLEF 2023

Solution Writeup · 4th place · Jun 6, 2023

First of all, thanks to Kaggle and Cornell Lab of Ornithology for hosting this interesting competition. And I would like to emphasize the thanks to the bird sound recordists who provided their data through xeno-canto.

Solution Summary
Knowledge Distillation is all you need.
Adding no-call data, xeno-canto data, and background audios(Zenodo) is effective.
Datasets
Additional datasets can be found here.

Bird CLEF 2023
Bird CLEF 2021, 2022 (for pretraining)
ff1010bird_nocall (5,755 files for learning no call)
xeno-canto files not included in training dataset CC-BY-NC-SA (896 files) and CC-BY-NC-ND (5,212 files).
Zenodo dataset (background noise)
esc50 (rain, frog)
aicrowd2020_noise_30sec (background noise for pretraining)
Models
My models are based on the BirdCLEF 2021 2nd place solution using timm.
My solution consists of a total of 4 models, all with eca_nfnet_l0 as the backbone. Each set is slightly different.
MelSpectrogram (sample_rate: 32000, mel_bins: 128, fmin: 20, fmax: 16000, window_size: 2048, hop_size: 512, top_db=80.0, NormalizeMelSpec) following this solution.
Public: 0.8312, Private: 0.74424
Balanced sampling with the above settings.
Public: 0.83106, Private: 0.74406
PCEN (sample_rate: 32000, mel_bins: 128, fmin: 20, fmax: 16000, window_size: 2048, hop_size: 512).
Public: 0.83005, Private: 0.74134
MelSpectrogram (sample_rate: 32000, mel_bins: 64, fmin: 50, fmax: 14000, window_size: 1024, hop_size: 320).
Public: 0.83014, Private: 0.74201
Training
Knowledge distillation of pre-computed predictions in Kaggle Models (bird-vocalization-classifier) was the hallmark of my solution. This model cannot complete inference in less than 2h, but it is very powerful. cmAP_5 on my validation dataset was 0.9479. So I tried to extract useful information from this model.
The Kaggle models pre-computed predictions were created here.
According to this paper, they argued that efficient distillation required 1. consistent input, 2. aggressive mixup, and 3. a large number of epochs. So I did 2 and 3 because it was challenging to integrate the Kaggle model into my training pipeline. It certainly looked effective.
I chose only approaches that improve both CV and LB. In this competition, I just couldn't believe in one or the other.

Using 5 StratifiedKFold. Only 1 fold did not cover all classes, so the remaining 4 were used for training.
The evaluation metric is padded_cmap1. The best results were the same with padded_cmap5 in most cases.
Use primary_label only
no-call is represented by all 0
loss: 0.1 * BCEWithLogitsLoss (primary_label) + 0.9 * KLDivLoss (from Kaggle model)
Softmax temperature=20 was best (tried 5, 10, 20, 30).
I used randomly sampled 20-sec clips for training. Audio that is less than 20 sec is repeated.
epoch = 400 (Most models converge at 100-300)
early stopping(pretraining=10, training=20)
Optimizer: AdamW (lr: pretraining=5e-4, training=2.5e-4, wd: 1e-6)
CosineLRScheduler(t_initial=10, warmup_t=1, cycle_limit=40, cycle_decay=1.0, lr_min=1e-7, t_in_epochs=True,)
mixup p = 1.0 (It was better than p=0.5)
I also encountered a situation where the training time was significantly longer when pretrained with past competition data. Thanks to everyone who suggested solutions.

Augmentation
OneOf ([Gain, GainTransition])
OneOf ([AddGaussianNoise, AddGaussianSNR]
AddShortNoises esc50 (rain, frog)
AddBackgroundNoise from Zenodo. The 60 minutes with the fewest bird calls were extracted from each dataset and divided into 30 sec (training only).
AddBackgroundNoise from aicrowd2020_noise_30sec and ff1010bird_nocall (pretraining only).
LowPassFilter
PitchShift
Hardware
1 * RTX 3090
Pretraining time: 30-40h per model.
Training time: 9-12h per model.
Inference
Reducing the inference time is the part I was having trouble with. Thanks to @leonshangguan for sharing his effective approach, I was able to use 4 models.

Ensemble with simple averaging.
4 models using PyTorch JIT. total 110min.
Inference notebook
What didn’t work
focal loss.
Split a low-sample class.
Backbone other than eca_nfnet_l0 and eca_nfnet_l1.
Optimizers (adan, lion, ranger21, shampoo). I tried to create a custom normalize-free model but failed.
CMO (mixup worked better when using distillation).
CQT (slow and degraded).
Change to first stride(1, 1). CV was good but inference takes a long time and not a single model is completed in less than 2h.
Pretraining Zenodo data (story before trying distillation).
Distillation training from scratch (not use imagenet weights). It didn't converge in the amount of time I could tolerate.
MelSpectrogram, PCEN, and CQT integrated into input channels in all combinations. There was no synergistic effect.
Ablation study
Name	Public LB	Private LB
BaseModel	0.80603	0.70782
BaseModel + Knowledge Distillation	0.82073	0.72752
BaseModel + Knowledge Distillation + Adding xeno-canto	0.82905	0.74038
BaseModel + Knowledge Distillation + Adding xeno-canto + Pretraining	0.8312	0.74424
BaseModel + Knowledge Distillation + Adding xeno-canto + Pretraining + Ensemble (4 models)	0.84019	0.75688
Acknowledgments
My solution builds on contributions from participants in this and past competitions, birding enthusiasts, and the work of the machine learning community. thank you very much 🙏
Training Code: https://github.com/AtsunoriFujita/BirdCLEF-2023-Identify-bird-calls-in-soundscapes

Project Details
Author
atfujita
atsunorifujita


Share
3
19 Comments
Hotness
Pardeep Singh
Posted 2 years ago

447th in this Competition

Congrats @atsunorifujita and thanks for sharing this detailed writeup 🙏


Reply
Jan H.
Posted 2 years ago

57th in this Competition

Congratulations, and thanks for the helpful post! Could you also make your training code available on github and post a link?


Reply
atfujita
Topic Author
Posted 2 years ago

4th in this Competition

Absolutely! I'm preparing for it.


Reply
yukiya
Posted 2 years ago

70th in this Competition

Thank you very much in advance. I am interested in distillation, but I cannot make it work


Reply

3 more replies
Profile picture for atfujita
Profile picture for yukiya
Profile picture for Jan H.
Jan Schlüter
Posted 3 years ago

72nd in this Competition

Congratulations, and thanks for sharing! Just to see if I got it correctly, your training steps were:

Take a backbone that is pretrained on ImageNet
Train a model with this backbone on BirdCLEF 2021+2022+ff1010, with Google's bird vocalization classifier as a teacher, lr 1e-5
Replace the last layer, train the model further on 2023+ff1010+xeno-canto, again with the same teacher, lr 2.5e-4
Any idea how much you gained in inference time from using PyTorch's JIT? I tried PyTorch 2's compilation on single models or the full ensemble, but it didn't help at all (probably also because of graph-breaking Python control statements).


Reply
atfujita
Topic Author
Posted 3 years ago

4th in this Competition

Thank you!

My training steps are as you said. To be precise, ff1010 in 2. is used as background noise.

PyTorch JIT reduced the inference time by about 20%.
I also tried ONNX, but didn't use it because of the complexity of the inference pipeline. The PyTorch JIT was able to include MelSpectrogram and PCEN in the model and converted it, simplifying the inference code.


Reply
Aaditya Bansal
Posted 3 years ago

284th in this Competition

Thank you for sharing your approach, I am a beginner in Deep Learning, do you have any tips as to how I can approach complex problems like this.


Reply
atfujita
Topic Author
Posted 3 years ago

4th in this Competition

Thank you, I think we just have to come up with ideas and experiment. That's why we always have to keep learning.


Reply
Gaurav Srivastava
Posted 3 years ago

233rd in this Competition

Awesome! Congratulations @atsunorifujita 🔥


Reply
atfujita
Topic Author
Posted 3 years ago

4th in this Competition

Thank you 👍


Reply
Nischay Dhankhar
Posted 3 years ago

9th in this Competition

Congrats on a great solo finish @atsunorifujita 🎉 your distillation technique is indeed powerful.


Reply
atfujita
Topic Author
Posted 3 years ago

4th in this Competition

Thank you! I hope you are enjoying your work 👍


Reply
Harshit Sheoran
Posted 3 years ago

9th in this Competition

Congratulations on your solo gold! Great Find on how to make use of already existing model


Reply
atfujita
Topic Author
Posted 3 years ago

4th in this Competition

Thank you and congratulations 🙌


Reply
Leon
Posted 3 years ago

10th in this Competition

Congrats for your solo gold!💯


Reply
atfujita
Topic Author
Posted 3 years ago

4th in this Competition

Thank you and congratulations on your solo gold 🎉


Reply
Pushkar Ambastha
Posted 2 years ago

Congratulations and brilliant discussion. A few questions since I saw the ps recently:

Is there any reason why Focal Loss cannot outperform BCE?
Why do you prefer librosa over torchaudio?
Is there any substantial postprocessing you used in inference or that you plan to use as future work?
What models are appropriate for ONNX?
How do you get started with PyTorch jit? Do you have any recommended documentation?

Reply
atfujita
Topic Author
Posted 2 years ago

4th in this Competition

Thank you and nice questions!

Is there any reason why Focal Loss cannot outperform BCE?
I don't know why. I was just comparing the experimental results. Focal loss may be better for different data.

Why do you prefer librosa over torchaudio?
Because I wanted my model inputs and KaggleModel inputs to be as consistent as possible. Firstly I was using soundfile.

Is there any substantial postprocessing you used in inference or that you plan to use as future work?
I can't think of anything at the moment. A simple average of multiple models worked.

What models are appropriate for ONNX?
It was possible to convert to ONNX by separating mel spectrogram and PCEN from my model. However, I got lost motivation to use it because the inference time was not significantly different from PyTorch JIT.

How do you get started with PyTorch jit? Do you have any recommended documentation?
In many cases, it's pretty straightforward (it fails with unsupported layers). The official documentation is helpful, but you can convert it with a few lines of code.

https://pytorch.org/docs/stable/jit.html


Reply
Pablo Larrosa
Posted 2 years ago

530th in this Competition

Congratulations!, I'd like to know how many time you needed to train your folds in a RTX 3090.

Thank you in advance!


Reply
atfujita
Topic Author
Posted 2 years ago

4th in this Competition

Thank you and good question!

Pretraining time: 30-40h per model.
Training time: 9-12h per model.

Reply
Pablo Larrosa
Posted 2 years ago

530th in this Competition

How much memory the RTX 3090?


Reply
atfujita
Topic Author
Posted 2 years ago

4th in this Competition

RTX 3090 has 24GB of VRAM


Reply
Skip to
content
Kaggle

Create
Search

Home

Competitions

Datasets

Models

Benchmarks

Game Arena

Code

Discussions

Learn

More


View Active Events

Skip to
content
Kaggle

Sign In

Register
5th place solution

BirdCLEF 2023

Solution Writeup · 5th place · Jun 8, 2023

First of all, thanks to the Cornell Lab of Ornithology and the Kaggle Team for hosting this competition. It was a great opportunity to learn something new.

In this post, I want to present a summary of my solution.

Datasets
2023/2022/2021 competition data
Additional Xeno-Canto data containing 2023 comp. species in foreground and background
ESC50 noise
No-call noise from 2021 competition data
Models
I used SED architecture (same as in @tattaka 4th place solution) with following backbones:

tf_efficientnet_b1_ns
tf_efficientnet_b2_ns
tf_efficientnet_b3_ns
tf_efficientnetv2_s_in21k
Training
I trained all models in two steps:

Pretrain with 2022/2021 data. I used only white noise (p=0.5) for this step
Finetune on 2023 data with the following augmentations:
For waveform - Mixup (p=1) and OneOf([White noise, pink noise, brown noise, noise injection, esc50 noise, no-call noise]) (p=0.5)
For spectrogram - Two time masks (p=0.5 each) and one freq mask (p=0.5)
Training details:

All models were trained on 5-sec clips
4-fold stratified CV split
I used both primary and secondary labels
BCEWithLogitsLoss with weight for each sample based on the rating
AdamW - 5e-4 lr, 1e-3 weight decay for most of the models
CosineLRScheduler with default parameters
40 epochs - the best score was almost always in the last epoch, so in addition to 4 folds, I also trained the 5th model using all available data. This full-fit version was consistently better than one-fold by 0.002-0.003 public and private LB.
Some models were finetuned on soft/hard pseudolabels.
My best model was trained with a combination of actual labels for competition data and hard pseudo labels for XC data. It has 0.72714/0.81836 private/public LB and 10 minutes submission time.

Inference
The probabilities from 15 models (folds) were averaged. 6 of these models have the same first three layers of the backbone, so the number of models is not very fair)

I've also used ONNX runtime and ThreadPoolExecutor - thanks to the @leonshangguan for sharing this notebook.

Inference kernel: https://www.kaggle.com/evgeniimaslov2/birdclef-5th-place-code
Github: https://github.com/yevmaslov/birdclef-2023-5th-place-solution

Project Details
Author
Yevhenii Maslov
evgeniimaslov2


Share
7 Comments
Hotness
Awais Asghar
Posted 2 years ago

Congratulation dear, many thanks for share details


Reply
Pardeep Singh
Posted 2 years ago

447th in this Competition

Congrats @evgeniimaslov2 and thanks for sharing this writeup! 🙏🎉


Reply
Hakan Dogan
Posted 2 years ago

669th in this Competition

Hi! Many thanks for the details of your solution. I have a question on training "using all available data". When you had no validation data, did you have a specific criteria for early stopping or did you still train all 40 epochs?


Reply
Yevhenii Maslov
Topic Author
Posted 2 years ago

5th in this Competition

Hi! I trained all 40 epochs


Reply
Jan H.
Posted 2 years ago

57th in this Competition

Thanks - that's very helpful! Any chance you could also put your training code on github and post a link to that?


Reply
Yevhenii Maslov
Topic Author
Posted 2 years ago

5th in this Competition

Hi! Sure, I'll update this thread with a link once I've finished cleaning up the code.


Reply
Yevhenii Maslov
Topic Author
Posted 2 years ago

5th in this Competition

Hi @janhuus I've uploaded the training code to github, hope it's still relevant to you


Reply
Skip to
content
Kaggle

Create
Search

Home

Competitions

Datasets

Models

Benchmarks

Game Arena

Code

Discussions

Learn

More


View Active Events

Skip to
content
Kaggle

Sign In

Register
6th place solution: BirdNET embedding + CNN

BirdCLEF 2023

Solution Writeup · 6th place · May 25, 2023

Thank you to the host and Kaggle for organizing the competition.

My solution is a combination of the embedding vectors of BirdNET-Analyzer V2.2 and the CNN from the BirdCLEF 2021 2nd place solution.

Model (BirdNET embedding + CNN)
In order to utilize the features of BirdNET, the embedding of BirdNET is concatenated to the output of the CNN of the BirdCLEF 2021 2nd place solution.



BirdNET embedding
The embedding vectors of BirdNET were created with BirdNET V2.2 embeddings.py. I modified BirdNET V2.2 audio.py so that if the length of the audio data is shorter than BirdNET's sample rate (48000), the data is padded to output at least 1 second of embedding vector. Using V2.2 rather than the latest version BirdNET V2.3, slightly improved cv.

CNN
The backbone of the CNN from the BirdCLEF 2021 2nd place solution used timm's eca_nfnet_l1 and seresnext26t_32x4d.

Training
After pretraining with data from BirdCLEF 2021 + BirdCLEF 2022, I train the CNN and other linear layers with data from BirdCLEF 2023. BirdNET is not trained.

The input for training is data for 30 seconds. Since BirdNET outputs embedding vectors for 3 seconds of data, I averaged each of the embedding vectors for 30 seconds.

In most experiments, cv was highest in the final epoch, so I included all data in the training set.

The main training parameters are as follows:

loss: BCEWithLogitsLoss

MelSpectrogram

sample_rate: 32000, window_size: 1024, hop_size: 320, fmin: 0, fmax: 14000, mel_bins: 128, power: 2, top_db=None
labels: primary label=0.9995, secondary label=0.4 or 0.5 (BirdCLEF 2022 3rd place solution), label smoothing=0.01

cv: primary_label StratifiedKFold(n_splits=5)

optimizer: AdamW(weight_decay=1e-4)

scheduler: warmup 0-3epoch(lr=3e-6->3e-4) + Cosine Annealing 3-70epoch(lr=3e-4->3e-6)

Augmentation
As in the previous two competitions, augmentation was important. I combined the following augmentations:

audiomentations.Shift
Mixup (BirdCLEF 2021 2nd place solution)
Gaussian Noise
random lowpass filter
random power (BirdCLEF 2021 5th place solution)
torchaudio.transforms.FrequencyMasking
torchaudio.transforms.TimeMasking
Oversampling
The following Oversampling was performed, but I think the effect was minimal.

Oversampling to have at least 20 of data for each class

I doubled the number of training data that have two or more secondary labels because data with more than two secondary_labels have worse val_loss

As in the BirdCLEF 2022 5th place solution, files with reverb/echo effects created for some minority class data and added to the training data

Strong augmentation was applied to the oversampled data

Inference
The CNN input is given 5 seconds of data for inference, and the BirdNET input is given the first 3 seconds of data for inference.

To submit within 2 hours, I converted each model to ONNX

As in this notebook, I also used ThreadPoolExecutor to speed up inference

Ensembling
The outputs of 2 models with different CNN backbones (eca_nfnet_l1, seresnext26t_32x4d) simply averaged.

What did not work
add background noise (BirdCLEF 2021 2nd place solution)
postprocess (BirdCLEF 2022 2nd place solution)
Speeding up inference using OpenVINO
Even if the model was converted to OpenVINO, the inference time was not much different from ONNX. I think I just used it wrong.
Using MFCC as input (as in this paper)
Using CQT as input
Using FAMILY etc. from eBird_Taxonomy_v2021.csv for prediction (as in this paper)
Rough lb history
name	Private Score	Public Score
BirdCLEF 2021 2nd place solution CNN (with BirdCLEF 2021 + BirdCLEF 2022 pretrain)	0.71	0.80
BirdCLEF 2021 2nd place solution CNN + Augmentation + BirdNET embedding	0.74	0.82
BirdCLEF 2021 2nd place solution CNN + Augmentation + BirdNET embedding + Oversampling + Ensembling	0.75	0.83
Project Details
Author
anonamename
anonamename


Share
10 Comments
Hotness
Pardeep Singh
Posted 2 years ago

447th in this Competition

Great work @anonamename and congrats on the solo gold. Thanks for sharing the writeup 🎉


Reply
Ayush Jha
Posted 3 years ago

Congrats on getting 6th position and for getting gold!!!


Reply
Nischay Dhankhar
Posted 3 years ago

9th in this Competition

Great way to integrate BirdNet, congratulations on the gold! 🎉


Reply
Aphysict
Posted 3 years ago

12th in this Competition

Congratulations and thanks for sharing! I remember that BirdNET is trained on sample rate 48000 audios, how do you handle the difference on sample rate?


Reply
anonamename
Topic Author
Posted 3 years ago

6th in this Competition

The sample rate has not been changed. The sample rate was left at its default, which resamples to 48000, because I thought that changing and loading the sample rate to 32000 would not be suitable for BirdNET's input data.


Reply
atfujita
Posted 3 years ago

4th in this Competition

Congratulations and thanks for sharing! The results for only 2 models were impressive. Better than my single model😨


Reply
Aaditya Bansal
Posted 3 years ago

284th in this Competition

Thank you for sharing your work, as a beginner just getting started with Deep learning I am amazed with how you can figure out so many details and approaches to solve a problem. Do you have any tips for how I can get started to efficiently solve these complex problems?


Reply
anonamename
Topic Author
Posted 3 years ago

6th in this Competition

I still don't know how to solve these problems efficiently. In my case, I carefully examine the data, think of hypotheses that might solve the problem, and experiment while gradually improving the model hundreds of times. It's through this process of trial and error that a good solution is found, albeit very rarely.


Reply
Aaditya Bansal
Posted 3 years ago

284th in this Competition

Understood😊 I will keep on learning and experimenting and once again thank you for sharing your approach.


Reply
Harshit Sheoran
Posted 3 years ago

9th in this Competition

Congratulations!, We tried something similar with BirdNet, great to see that you made it work!


Reply
Gaurav Srivastava
Posted 3 years ago

233rd in this Competition

Congratulations 🎉 And really thanks for sharing


Reply
Skip to
content
Kaggle

Create
Search

Home

Competitions

Datasets

Models

Benchmarks

Game Arena

Code

Discussions

Learn

More


View Active Events

Skip to
content
Kaggle

Sign In

Register
top 7th solution - `sumix` augmentation did all the work

BirdCLEF 2023

Solution Writeup · 7th place · May 28, 2023

What's important in our best inference notebook ?
Ensemble of 19 models
Ensemble of efficientnet_b2 from torchvision and rexnet150 from timm
Speedup with openvino
For submission we trained all models with knowledge distillation on full dataset - (rkl prefix is indicator)
Our five models from 19 with rkl-rx-wdfbn-rx-ema-200-full prefix achieve 0.7471 on private test set and 0.83194 on public test set, rx - rexnet150, distillated rexnet150 models from rexnet150 trained before

Custom Augmentations
Let's imagine you're at some party right now and you're surrounded by talking people. It's not hard for you to pay attention to a certain group to hear their conversation and you can freely change your target. So the task - to cut specific speaker from audio - is natural. What we can say about audio in which - one person cries and another one laughs? In such audio there are two actions - laughing and crying, and this is true for birds too. With this consideration we've come up to sumup idea:

def sumup(waves: torch.Tensor, labels: torch.Tensor):
    batch_size = len(labels)
    perm = torch.randperm(batch_size)

    waves = waves + waves[perm]

    return {
        "waves": waves,
        "labels": torch.clip(labels + labels[perm], min=0, max=1)
    }
Boosting score
What if you have an audio where one bird is more louder than another? This audio is still with two different birds, but if signal from second bird is tiny, it's okay to assume, that maybe probability of second bird is small or even zero.

def sumix(waves: torch.Tensor, labels: torch.Tensor, max_percent: float = 1.0, min_percent: float = 0.3):
    batch_size = len(labels)
    perm = torch.randperm(batch_size)
    coeffs_1 = torch.rand(batch_size, device=waves.device).view(-1, 1) * (
        max_percent  - min_percent
    ) + min_percent
    coeffs_2 = torch.rand(batch_size, device=waves.device).view(-1, 1) * (
        max_percent  - min_percent
    ) + min_percent
    label_coeffs_1 = torch.where(coeffs_1 >= 0.5, 1, 1 - 2 * (0.5 - coeffs_1))
    label_coeffs_2 = torch.where(coeffs_2 >= 0.5, 1, 1 - 2 * (0.5 - coeffs_2))
    labels = label_coeffs_1 * labels + label_coeffs_2 * labels[perm]

    waves = coeffs_1 * waves + coeffs_2 * waves[perm]
    return {
        "waves": waves,
        "labels": torch.clip(labels, 0, 1)
    }
We propose yet another customization of data mix specific for audio domain - sumix augmentation. As you can see, we used linear slope in our experiments, but it's open question what's slope for labels better to use, maybe min_percent = 0.3 doesn't need slope at all. This augmentation is applicable to any audio classification task. sumix is better than sumup because it also affects loudness, i. e. implies another audio augmentation.

When we decided to check solutions from past competitions, we tried to add background noise in our pipeline, but noise didn't show any improvement. Our thoughts - incorporating new background noises for audios is already in sumix or sumup. For gaussian/colored noises we have similar explanation, why those augmentations didn't work - competition's data is not excellent and consists of noisy samples, so sumix and sumup propagate noise and make noise diverse by themselves.

Training pipeline
concatmix with probability 0.5 - implementation
sumix(min_percent=0.3, max_percent=1.0) with probability 1
mixup on mel-spectrograms with Beta(1.5, 1.5) with probability 1
cutmix on mel-spectrograms with Beta(1.5, 1.5) with probability 0.5
AdamW with constant LR 1e-3 and weight_decay 1e-2 for all models
binary_cross_entropy for every class, labels are primary labels plus secondary labels
Random 5 continuous seconds from audio at training, first 5 seconds at validation
We turned off weight_decay for all normalization layers
EMA with decay 0.999
Almost total correlation between LB and our CV, ensemble of 20 models gathered as best in CV achieves 0.75021 on private test set and 0.83328 on public test set, this ensemble is our top-3 submission on public test set
Knowledge distillation
After training a model, we use it as teacher for another model with the same architecture, augmentations, AdamW, EMA and etc., except loss. At second stage with distillation we use

Models
rexnet150 with dropout 0.3 and drop_path 0.2
rexnet150-time - same rexnet150 but with attention head, distilled from rexnet150 without time-aware head
efficientnet-b2 with dropout 0.3 and drop_path 0.2
efficientnet-b2-time - analogously, distilled from efficientnet-b2
Already mentioned hacks allow to reach score 0.7471 with 5 rexnet150 trained on full data with batch_size=336 200 epochs and knowledge distillation on a single A100 (40GB). To boost performance we ensemble rexnet150 with efficientnet-b2 from torchvision. To boost performance yet another time we use attention head - copy-past self-attention operation from transformers. All experiments were conducted in mix-precision with bfloat16 type.

Ensemble of 5 models with prefix rkl-rx150-wdfbn-rx150-time-ema-ep200-full-{5seeds}, i.e rexnet150 with attention head for time awareness distilled from simple rexnet150, achieves 0.75151 on private set and 0.83261 on public set.

We didn't try:
External data not from BirdClef2021/BirdClef2022
SSL
SED models
Utilization of BirdNet
eca_nfnet from timm
We tried, but it didn't work:
Pretraining our models on BirdClef2021/BirdClef2022, we made few attempts, it didn't show any improvements in LB, experiments conduction became much slower, so we gave up the idea
Any background noise (even which was used in previous year's competition)
Any LR scheduler
Any Noise(Gauss, Colored)
tf_efficientnet_b2_ns from timm
SwinV2/ConvNeXt from torchvision
Manifold MixUp
focal_loss, soft_focal_loss
Weighted samplers
SpecAugment
Speeding up inference time
We measured inference time with random input in kaggle kernel, I don't remember which one of intel's processors was, often in our kaggle notebook we got Intel Xeon 2.2 Gz.



Our thoughts - ONNX by itself utilizes 2 cpus, OpenVino is faster cause it additionally fuses some layers for inference and uses compilation specific to a certain CPU in submission notebook to take advantage of processor's type

Appendix: MultiHeadAttentionClassifier's implementation. All hyperparameters are preserved
class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        input_channel: int,
        head_size: int,
        num_heads: int,
        attention_dropout: float,
    ) -> None:
        super().__init__()
        hidden_dim = head_size * num_heads
        self.hidden_dim = hidden_dim

        self.head_size = head_size
        self.num_heads = num_heads

        self.key = nn.Linear(input_channel, hidden_dim)
        self.query = nn.Linear(input_channel, hidden_dim)

        self.attention_dropout = nn.Dropout(attention_dropout)
        self.sqrt_head_size = sqrt(head_size)

        self.value = nn.Linear(input_channel, hidden_dim)

    def tranpose_for_scores(self, x: Tensor) -> Tensor:
        # [BS; T; H] -> [BS; T; K, M]
        new_x_shape = x.size()[:-1] + (self.num_heads, self.head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3) # [BS; K; T; M]

    def get_key(self, x: Tensor) -> Tensor:
        return self.tranpose_for_scores(self.key(x))

    def get_query(self, x: Tensor) -> Tensor:
        return self.tranpose_for_scores(self.query(x))

    def get_value(self, x: Tensor) -> Tensor:
        return self.tranpose_for_scores(self.value(x))

    def forward(self, x: Tensor) -> Tensor:
        # [BS; L; C]
        key = self.get_key(x)
        query = self.get_query(x)
        value = self.get_value(x)

        attention_scores = torch.matmul(query, key.transpose(-1, -2)) 
        attention_scores /= self.sqrt_head_size
        attention_scores = F.softmax(attention_scores, dim=-1)
        attention_scores = self.attention_dropout(attention_scores)
        x = torch.matmul(attention_scores, value) # [BS; K; N; M]
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(x.shape[:2] + (self.hidden_dim,))
        return x

class MultiHeadAttentionClassifier(MultiHeadSelfAttention):
    def __init__(
        self,
        input_channel: int,
        head_size: int = 32,
        num_heads: int = 24,
        attention_dropout: float = 0.3,
        num_classes: int = 264,
        dropout: float = 0.3
    ) -> None:
        super().__init__(input_channel, head_size, num_heads, attention_dropout)
        self.query = nn.Parameter(
            torch.empty(num_heads, num_classes, head_size),
            requires_grad=True
        )
        nn.init.normal_(self.query)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Conv1d(num_classes, num_classes, kernel_size=self.hidden_dim, groups=num_classes)
        )

    def get_query(self, x: Tensor) -> Tensor:
        return self.query

    def forward(self, x: Tensor) -> Tensor:
        # x.shape == [BS; C; F; T] // output from backbone
        x = vec.view(x.shape[:-2] + (-1,))
        # [BS; C; L]
        x = vec.permute(0, 2, 1)
       # [BS; L; C]

        out = super().forward(x)
        return self.classifier(out).squeeze()
Project Details
Authors
tingir
loonypenguin

Danil Gaynanov
eviltsundera

Mihail Dremin
dreminm

slime
martynoveduard


Share
1 Comment
Hotness
Pardeep Singh
Posted 2 years ago

447th in this Competition

Congrats for landing in the gold zone and thanks for sharing the writeup


Reply
Skip to
content
Kaggle

Create
Search

Home

Competitions

Datasets

Models

Benchmarks

Game Arena

Code

Discussions

Learn

More


View Active Events

Skip to
content
Kaggle

Sign In

Register
8th Place Solution: Implementing Multimodal Data Augmentation Methods (last updated 05/28)

BirdCLEF 2023

Solution Writeup · 8th place · May 28, 2023

Congratulations to all the winners!
Our gratitude extends to Kaggle and the Cornell Lab of Ornithology for their exceptional efforts in organizing this intriguing competition. In particular, I was able to contribute to the improvement of accuracy with highly original methods, so I was able to enjoy the competition very much.
Allow me to give a succinct overview of my solution. I’ll provide a more detailed update on the solution in the following days.

Summary
In my solution, only one type of backbone is used, and multimodal data augmentation techniques such as object detection, recommendation systems, and NLP are used to increase diversity and improve model accuracy.

Preprocess Pipeline Summary
We built a preprocessing pipeline that prevents overfitting by devising preprocessing so that the combinations that are actually possible while maintaining the diversity of the data. In particular, in preprocessing for Sound Object Detection, weighting the signal density of the spectrum made it easier to pick out the areas where the contribution of secondary labels is included. Thanks to that, the training accuracy has improved even when the value of the secondary labels is large.


Training Pipeline
In order to grasp the features of bird calls with various periodicities, build a pipeline that first learns with a long spectrum input, then lowers the time axis direction for each epoch, and finally learns local features. Did. Preprocessing by SOD prevents noise-like offsets from being selected even for local spectra, so training is stable.


code detail: https://www.kaggle.com/code/kunihikofurugori/8th-solution-down-scaling-detail/notebook?scriptVersionId=131297676

MultiModal Data Augments Part(in Detailed)
Code detail: https://www.kaggle.com/code/kunihikofurugori/8th-solution-notebook/notebook

・Time Segment DownScaling Augmentation (Global to Local)
Regarding the mini-batch method represented by the 2nd place solution of birdclef2021, we trained to learn local features while pre-learning global features by training to evolve the number of mini-batches from 15 to 1 at each epoch.

By first learning the global features and then learning the local features, it becomes possible to predict the local features after understanding the global features.

・Weak to Strong Preprocessing by YOLOv8 (Sound Object Detection)
I use the data pointed out by the competition host to train the yolov8 object detection model, and select an offset that will increase the number of object detections as much as possible.
Also, by minimizing the bounding box and reliability-weighted IoU for the frequency masking, Frequency Masking effectively hides noise.

Sample Inference
Inferece Sample Sound Object Detection

Reference Notebook
[1]https://www.kaggle.com/code/kunihikofurugori/sound-object-detection-prepare-annot-data/notebook
[2]https://www.kaggle.com/code/kunihikofurugori/birdclef2023-soundobjectdetectiontrain/notebook
[3]https://www.kaggle.com/code/kunihikofurugori/sound-object-detection-batch-predict/notebook
[4]https://www.kaggle.com/code/kunihikofurugori/birdclef2023-soundobjectdetectionanalysis/notebook

・Various Mixup Strategy
By using various mix-up methods, we tried to make it impossible to choose unnatural pairs. Details of the method are described below.

・Geometrical Mixup
Assuming that geographically close samples should have similar background noise, we created a model that mixes up geographically close samples without learning the background noise.

If the distance matrix is properly constructed, the amount of calculation will be O(n^2), so the code is written so that the parts that match up to the second decimal place of the longitude and latitude are treated as the same group.

・Resonance Mixup (Inspired by collaborative filtering)
Choosing a pair that resonates easily as a mixup destination is important for creating a more natural sound. Implemented a mixup that preferentially selects bird samples that are likely to resonate based on the collaborative filtering idea of the recommender system.

・Inner mixup for geometrical clustering space
Groups that are geographically close and have the same label do not change the label characteristics even if they share inputs and mixup. Therefore, we clustered such groups and made it possible to perform Inputmixup freely.

Models
My models are based on the BirdCLEF 2021 2nd place solution using backbone, eca_nfnet_l0.

Dataset
Bird CLEF 2023
Bird CLEF 2022/2021/2020 (Pretraining)
ff1010bird_nocall
Zendo data (YOLOv8)
CV setup
I thought it would be difficult to create a multifold model and check the LB due to resource limitations, so I did a simple test-train split instead of out-of-fold.

For the test data, we built validation by splitting 30 seconds into 5 second intervals and bootstrap sampling to the same size as the LB dataset size (8400 samples).

Furthermore, we found that using only the data with the primary label showed behavior similar to LB, so we excluded the data with the secondary label from the validation and calculated.

Inference Kernel: https://www.kaggle.com/code/kunihikofurugori/8th-place-solution-inference-kernel
Github link: https://github.com/furu-kaggle/birdclef2023

Result
commit_id	single inference kernel	LB	PB	tag
4a57a984d1f4101b9f470095e6a6fae0a6cc5af2	link	0.82628	0.73777	ecanfnet_nmel128_best1
55557aea59976c5befc11cbcd077f22c1d8289c0	link	0.82475	0.73301	ecanfnet_nmel128_best3
ccc435294a9f5f9989f5865def93da0817c3b030	link	0.82786	0.73341	ecanfnet_nmel128_best2
c9e077c172368c31c2d1974e0288554ebd014b2f	link	0.82279	0.72918	ecanfnet_nmel64_best2
a3791c561544adcc37507d5c27ec98e9d1c603a7	link	0.82172	0.73223	ecanfnet_nmel64_best1
ea70ecc95de1625212465b8e0148bc996e0237f9	link	0.82149	0.72850	ecanfnet_nmel64_best3
ecanfnet_nmel128_best1~3(300 stride pred) + ecanfnet_nmel64_best1~3 ensemble	link	0.83616	0.75285	best_submission
Project Details
Author
furu-nag
kunihikofurugori


Share
2 Comments
Hotness
Donghui Zhang
Posted 2 years ago

531st in this Competition

Congratulations on winning the eighth place in the competition and winning a gold medal! Your plan has benefited me a lot. By the way, in your Github, what do the 6 folders represent? I would like to reproduce it according to your report. Let me show you the structure of the experiment so that I can understand your thinking more deeply.


Reply
furu-nag
Topic Author
Posted 2 years ago

8th in this Competition

@gentlezdh
Thank you for your appreciation. github
6 folders are linked to commit_id of Result. For example, 4a57a98…script represents the training script for 4a57a984d1f4101b9f470095e6a6fae0a6cc5af2. Please ask again if you have any questions.


Reply
Donghui Zhang
Posted 2 years ago

531st in this Competition

Thank you for your answer. I am looking forward to discussing your plan with you by email or just asking here. This is really helpful for an undergraduate.


Reply
Skip to
content
Kaggle

Create
Search

Home

Competitions

Datasets

Models

Benchmarks

Game Arena

Code

Discussions

Learn

More


View Active Events

Skip to
content
Kaggle

Sign In

Register
9th Place Solution: 7 CNN Models Ensemble

BirdCLEF 2023

Solution Writeup · 9th place

Thanks to the organizers for hosting BirdClef 2023 again this year and giving us a learning experience.

Heartfelt Congratulations to the winners and everyone who got to learn and experience.

I would also like to express my gratitude to my talented teammates @ivanaerlic and @nischaydnk

Training
We used efficientnet_b0, eca_nfnet_l0 and convnext_tiny architectures in our final submission.

The training was done in multiple rounds using datasets from all previous years. All models were trained on 5s clips

For eca_nfnet_l0 and convnext_tiny
Hop size 320, n_mels 64, image_size = 64x501
For efficientnet_b0
Hop size 320, n_mels 64, image_size (Bilinear image size increase) = 128x1002

The training was done in multiple rounds:

Round 1:

Hop size 512, n_mels 64, image_size = 64x313
(Bilinear image size increase) = 128x636

Models : Eca nfnet l0 (128x636), convnext tiny (128x636).

Training data: 2022 & 2023

Augmentations: HFlip, Random Cutout, Pink Noise, Gaussian Noise, Noise Injection, Random Volume

Type of Training: Competition Labels

CV: ~.79 (Low)

Round 2:

Hop size 320, n_mels 64, image_size = 64x501

Models : Efficientnet B0 (128x1002 Bilinear increase), Eca nfnet l0 (64x501), convnext tiny (64x501).

Augmentations: HFlip, Random Cutout, Pink Noise, Gaussian Noise, Noise Injection, Random Volume

Training data: 2022 & 2023

Type of Training: Distillation

CV: ~.79 (Slight increase)

Round 3:

Hop size 320, n_mels 64, image_size = 64x501

Models : Efficientnet B0 (128x1002 Bilinear increase), Eca nfnet l0 (64x501), convnext tiny (64x501).

Training data: 2021 & 2022 & 2023

Extra methods: AWP, SWA, Attention Head

Augmentations: HFlip, Random Cutout, Pixel Dropout

Type of Training: Competition Labels

CV: ~.8

Round 4:

Hop size 320, n_mels 64, image_size = 64x501

Models : Efficientnet B0 (128x1002 Bilinear increase), Eca nfnet l0 (64x501), convnext tiny (64x501).

Training data: 2023

Augmentations: HFlip, Random Cutout, Pixel Dropout, Pink Noise, Gaussian Noise, Noise Injection, Random Volume

Extra methods: SWA, Attention Head

Type of Training: Pseudo Labels

CV: ~.81 (mild leak)

Our CV was the first 5 seconds of the clips, this CV although was not always consistent with LB, gave us a good enough idea of what we could expect.

Pseudo labeling improved our score a lot. We used pseudo-labels to make new secondary labels and gave them a hard probability 0.5

For convnext model, it was first fine tuned to increase the score on the targets where secondary label exists, although this sacrificed the primary label score, this adds diversity to our ensembles

Inference
We used OpenVino to increase the inference speed of our models

We also saw that loading an audio and giving it in batches, the first batch of inference is especially slower than the rest, so we made it so that it loads 8 audios at once, makes a data loader for all 8 audios and then forward pass the model, doing this we increased the inference speed further by 20%.

So we were able to submit 7 models, 2 of efficientnet_b0 + 2 of eca_nfnet_l0 + 3 of convnext_tiny

We also noticed that sigmoid after weight averaging was more effective than sigmoid before weight averaging, also we were using softmax activation which gave a better score than sigmoid up until we reached top of .82 (public LB), after that sigmoid started to work better.

BirdNet
Our best private submission turned out to be when we ensembled our models with BirdNet, although it was like our top 10 submissions of public and did give us a lot of variance, we did not select this because we were a little afraid of it being a dice roll on private LB.

Post Processing
Looking at the last year’s competition, we tried out increasing the confidence of a class by a factor of 1.5 if any prediction of that class in the audio file is more than .96

This did work on a lot of submissions and was in fact amongst our top subs, but in the end, it did not play well on private LB and we did not select it because of the risk of overfitting the public LB.

What did not work
SED models
Q-Transform
More than 5s training
Rolling Mean with last and next 5s segment

More than 60 submissions of ours had timeout errors and on the last second day of the competition, 4 of 5 submissions had encountered timeout error while before that the chance of getting a timeout was only 20-30% in the same notebook.

Project Details
Authors
Harshit Sheoran
harshitsheoran

Nischay Dhankhar
nischaydnk

Ivan Aerlic
ivanaerlic


Share
7 Comments
Hotness
Pardeep Singh
Posted 2 years ago

447th in this Competition

Great work team and thanks for sharing the writeup!


Reply
RihanPiggy
Posted 2 years ago

2nd in this Competition

Congratulations and thanks for sharing!
It is nice to see the birdnet worked!
Would you mind to share the code of converting eca_nfnet_l0 to openvino?
I had trouble in converting so I gave up using eca_nfnet_l0, big thanks!


Reply
Harshit Sheoran
Topic Author
Posted 2 years ago

9th in this Competition

Sorry for a very late reply, I somehow missed this comment

https://www.kaggle.com/code/harshitsheoran/openvino-opt/notebook

I think it was that ScaledStdConv2d was using batch norm and the default was going for training=True, which was causing error, so I remade the function and replaced the layers


Reply
Gaurav Srivastava
Posted 3 years ago

233rd in this Competition

Congratulations to you and thanks for sharing


Reply
Aphysict
Posted 3 years ago

12th in this Competition

Congratulations and thanks for sharing! What is the private LB score for single BirdNET ? I am wondering if I use BirdNET correctly.


Reply
Harshit Sheoran
Topic Author
Posted 3 years ago

9th in this Competition

A lot of people on Public .82 will get to .83 if they use BirdNet, I think it would be a 0.005 increase on average between both public and private… atleast that is what we observed


Reply
Aphysict
Posted 3 years ago

12th in this Competition

Thanks for the elaboration, I am actually asking about BirdNet performance alone, not the ensemble. Do you retrain the BirdNet or directly use BirdNet and average results?


Reply

4 more replies
Profile picture for Gilles Vandewiele
Profile picture for Aphysict
Profile picture for Harshit Sheoran
atfujita
Posted 3 years ago

4th in this Competition

Congratulations! Here again, OpenVino. Speeding up inference is a great learning experience for me.


Reply
Skip to
content
Kaggle

Create
Search

Home

Competitions

Datasets

Models

Benchmarks

Game Arena

Code

Discussions

Learn

More


View Active Events

Skip to
content
Kaggle

Sign In

Register
10th place solution

BirdCLEF 2023

Solution Writeup · 10th place · Mar 4, 2024

Training Data
I use 2023 competition data only, but 3 of my models have pre-train weights on 2022 data.

Model Architecture
I use SED architecture, almost the same as this notebook with very minor modifications (e.g., emb_size, num_layers).

I tried 8 backbones in total, including 3 large and 5 small:
large backbones: efficientnetv2_m, eca_nfnet_l0, seresnext50
small backbones: efficientnet_v0, mnasnet_100, spnasnet_100, mobilenetv2_100, resnet34

As seresnext50 and resnet34 perform not well based on my cv, so I dropped them in the final ensemble.

I got the information of mnasnet_100, spnasnet_100, and mobilenetv2_100 by asking ChatGPT: What are small models offered by timm and ChatGPT replied with these models with the number of parameters, which is very helpful.

Code
One of my inference notebooks can be found here

Training Strategy
all models are trained on 5s clip.

soft label: in the one-hot gt, I give the primary label with "1" and the secondary label with "0.3".

data augmentation: add GaussianNoise, Background noise, PinkNoise, adjust Gain, use cut mix and mix up (very helpful).

Progressive pretraining: the hardest part of this competition is to find a good set of parameters when applying melspectrum, especially the n_fft, hop_length, and n_mels, which affect the input image (melspectrum) size. I tried many different params, and surprisingly, I found that if you have a model trained on params set A, then you load the model as a pre-trained model (except the first and last layer), and then train it on params set B, the model will perform much better than directly training on params set B, and this is the key idea of my solution. All of my models were trained on many different parameters and finally finetuned on n_mels=128, n_fft=1024, and hop_length=320 to make it the same size, so that can be ensemble efficiently.

loss function: BCEWithLogits + BCEFocal2WayLoss, same as previous years.

cv strategy: I use 5fold stratified cross-validation and select the best 1 or 2 fold according to cv.

ensemble strategy: each model has 1/2 fold for ensemble, and I have 6/7 models in total, with 6 different backbones as mentioned before. The reason I didn't use all folds is because of the computational limitation asked by the host.

cpu acceleration: I use ThreadPoolExecutor as I have mentioned here, and I also convert models to onnx which utilizes the model quantization technique and shows faster inference time.

Others
more than 70 of my submissions encountered timeout error, and I therefore highly request the function that if more information can be released for the submission, for example, the running time @herbison . I suffered from it a lot, and many almost the same submissions sometimes pass but sometimes not, even with the identical code.
I previously made a request regarding the submission time function, as mentioned here, when @ammarali32 and @jaafarmahmoud1 found 10 PM to 6 AM EST is a good time to make submissions. The reason behind this recommendation is that 10 PM EST is considered late for the American region, while it corresponds to 4 AM in Europe and 10 AM in Asia, which might be too early for many participants. Consequently, this particular timeframe becomes somewhat magical, as very few people tend to make submissions during this period. Therefore, if one chooses to make a submission during this time, they could potentially benefit from it.

based on my experiments, I believe the model params have an impact on inference time when using CPU, i.e., with the same model architecture and the number of params, but different models, some may be fast some may not. @chrisqiu 's post verifies this situation. As the host is trying to build apps that can be deployed in the real world using the CPU, I think this issue is worth mentioning @stefankahl.

I found that both my CV and lb vary based on the different folds, which indicates a good data split may have a great impact on the result, and some luck is needed when setting the random seed for the data split.

Acknowledgement
I appreciate @stefankahl for hosting this amazing competition, @herbison for answering my questions regarding the submission timeout error, and thanks to the Kaggle community for sharing their insights, ideas, and thoughts. This is a wonderful learning experience and I hope everyone enjoys it!

Project Details
Author
Leon
leonshangguan


Share
2
8 Comments
Hotness
Maximiliano Diaz Battan
Posted 3 years ago

75th in this Competition

I'm really glad you could reach a gold medal, your ensemble notebook help us a lot, was one of the reasons why we reach a medal. Thanks for sharing your solution, we also thought Onnx could improve the inference time but sadly we couldn't make it work, thanks for sharing the inference code also, surely can help a lot of kagglers (including us) in future competitions.


Reply
Muhammad Bilal Hussain
Posted 3 years ago

@leonshangguan thanks for sharing with us your 10th place solution 🙌🏼🙌🏼


Reply
atfujita
Posted 3 years ago

4th in this Competition

Congratulations and thanks for sharing (include inference code)!


Reply
Leon
Topic Author
Posted 3 years ago

10th in this Competition

Thanks and congrats for winning the Prize!


Reply
Nooruddin Hyderabadwalla, CFA
Posted 3 years ago

Thanks for sharing your solution @leonshangguan. The discussion title is misleading in a way because it suggests you somehow had ChatGPT work for you, but you actually intelligently used it to an extent and still worked hard on this as can be seen by 70 submissions being timed out comment of yours. I'm getting to learn so much from so many people here, it's amazing! God bless!


Reply
Gaurav Srivastava
Posted 3 years ago

233rd in this Competition

Wow Crazy ! Thanks for sharing and Congratulations🎉


Reply
jionie
Posted 3 years ago

Congrats for your first solo gold! 💯


Reply
Leon
Topic Author
Posted 3 years ago

10th in this Competition

Thank you!!!


Reply
Skip to
content
Kaggle

Create
Search

Home

Competitions

Datasets

Models

Benchmarks

Game Arena

Code

Discussions

Learn

More


View Active Events

Skip to
content
Kaggle

Sign In

Register
1st place solution

BirdCLEF 2024

Solution Writeup · 1st place · Mar 10, 2025

Written by me and @arsenypoyda.

Brief introduction about happiness!
First of all, we want to thank the BirdCLEF Host and Kaggle Team for this competition. Two days ago, we found out that we were in 1st place, and it was unbelievable. Of course, it’s just luck, as the difference between us and 2 or 3 places is negligible, but nevertheless. Moreover, that day, we not only became Kaggle masters but also received a master's degree from the university (double masters 🙂).

Overview
Data/labels preprocessing.
BirdCLEF 2024 train_audio
pseudo labeled unlabeled_soundscapes
For the final submissions, we use only 2024 data, both train_audio and unlabeled_soundscapes.
At the very beginning of the competition, we found that random fold0 gives much better results than other folds. To find out why this happens, we calculated various statistics related to signal strength (see the picture below) and found that fold0’s statistics are lower than other folds’.

So for ensembles, instead of folds0-4, we use fold0 and 0.8 quantile of statistics T = std + var + rms + pwr, and this worked well. Seemingly, noisy and too loud audio harms the models.

There are some duplicate audio files in the train data, so we discard them. Data in train_audio is also filtered with Google-bird-vocalization-classifier: if the classifier’s max prediction doesn’t match with the primary label, the chunk is dropped (maybe there is no bird sound or it has a bad quality). If the classifier’s max prediction matches with the secondary label, we replace the primary label with the secondary label. Moreover, if the file has secondary labels, then we take the primary label with 0.5, and the remaining 0.5 is evenly distributed among the secondary labels. We also add pseudo labels obtained with Google classifier to the resulting labels with a coefficient of 0.05. The soundscapes are labeled with an ensemble of Google classifier, and our best models trained only with train_audio. Finally, if the sound is too short, we use cyclic padding.

Model input
Models are trained on 10-second chunks that consist of two 5-second adjacent chunks with averaged labels. The idea is that a 10-second chunk provides processing of full chirps or full periods of chirps (if they are cropped by 5-second chunks).

Mel parameters (10 seconds -> 1x128x640):
n_fft = 1024
hop_length = 500
n_mels = 128
fmin = 40
fmax = 15000
power = 2
Models
efficientnet_b0 pretrained on ImageNet
regnety_008 pretrained on ImageNet
We tried other models:
seresnext gives the same results as efficientnet, but the inference time is almost 3 times longer.
We saw that 3rd place (NVBird) uses efficientvit due to its high speed. In our case, ViTs work significantly worse.
Modifications like CNN from BirdCLEF 2021 2nd place solution and SED work slower and provide worse results than pure backbones. We are convinced that overly complex models do not work better than simple ones.
We don't experiment with larger models since we have no computation resources and use only Kaggle kernels.

Training!
Parameters:
CrossEntropyLoss
AdamW
CosineAnnealingLR scheduler
Initial learning rate 1e-3…3e-3
7–12 epochs
Batch size = 96
Training time of one model on Kaggle kernel P100 takes up to 2 hours.
Data augmentation:
random audio segment
XY masking
horizontal cutmix
First of all, we use only CE loss, not BCE (BCE shows significantly worse results than CE). It can be connected with the specificity of train data. There are too many labels (182), and almost always, only one of them is present (with augmentations up to 2-3). So, the problem is reduced to a multiclass one. During the training, CE loss leads to the multiclass problem, and logits are passed through SoftMax. However, in the inference, we do not use softmax and pass logits through sigmoid (postprocessing is discussed in the next section in detail).

Second, we use audio segmentation. During the epoch, the model sees only one random chunk from each file. The problem is that the train_audio has many small files (+ some really large ones), and unlabeled_soundscapes have few large files. So, we divide the audio into X-second segments that are considered as separate files. We tried X = {20, 30, 60}, which led to a different number of epochs: the smaller the X, the fewer the number of epochs (because the number of steps per epoch is greater). As a result, we balance the train_audio and unlabeled_soundscapes.

Postprocessing
To predict chunk n, the models take 10 seconds: 5 seconds from the chunk n and 2.5 seconds from the previous and next chunks.


Although logits are passed through softmax during the train, we use sigmoid in the inference. The two most important things in the inference pipeline are chunks averaging and ensembling with min() reduction. In the figure below, we demonstrate the best pipeline for each class in chunk n.


Using the sigmoid with CE-trained models leads to the predictions being noisy, so min() just lowers uncertain predictions.


Inference time optimization
Compilation with OpenVINO (fixed model input)
Parallel mel spectrograms computing with joblib
Storing all the computed mel spectrograms in RAM
As a result, one model processes the entire test data in ~18 minutes on Kaggle CPU kernel. An ensemble of 6 models, taking into account the creation of mel spectrograms, takes ~2 hours.
What didn't work
everything else…

Negligible change in score
183 “nocall” class
We add nocall samples from external data to the train and use an additional 183 class. For inference, we take only 182 classes with bird calls. There was no improvement in the public score. Now, we observe a significant increase in private score (0.655 -> 0.671)…
Mel spectrogram normalization
Other mel spectrogram parameters
Input image size 224x224
15 second chunk as model input
Softmax temperature
Weight averaging (SWA and EMA)
Pseudo labeling with softmax temperature
Pretraining on the previous years data
Noticeable decrease in score
BCEloss, BCEloss with positive weights, focalloss
Other augmentations (mixup, noise, pixdrop, blur, audio 1d augmentations, horizontal flip)
STFT instead of mel spectrogram
Additional data from Xeno-canto
We tried different approaches to improve the quality of additional data, such as filtering with Google classifier or BirdNet, taking random fold, and taking some quantile of statistics T. The best solution is not to use additional data… The private score also proves it.
Pseudo labeling of train data with high coefficient
It seems that pseudo-labeling for train_audio is a way of label smoothing, so it is better to use a small coefficient.
Train on 10 second chunks and inference on 5 second chunks
Train additional models to detect bird calls
The scores for such models are close to random/constant prediction
Multistage training
We tried to train several epochs on the train_audio and then on the unlabeled_soundscapes and vice versa. We also tried to train on the whole data and finetune on the train_audio or unlabeled_soundscapes.
Main steps to success
The table below shows changes relative to the baseline (our 1st submission) pipeline that gives noticeable improvement to the score.

baseline
efficientnet_b0
CEloss
softmax for the inference
first 5 seconds from each file
fold0 out of train_audio without duplicates
Main steps	Private score	Public score
baseline	0.544028	0.599798
sigmoid for the inference	0.588338	0.628777
random 5 second chunk	0.601803	0.638572
XY masking	0.601909	0.639358
horizontal cutmix	0.615368	0.670460
pseudo labeled unlabeled_soundscapes	0.639777	0.687000
60 second segmentation	0.649936	0.691752
secondary labels	0.642781	0.695215
filtering train_audio chunks with Google classifier	0.655190	0.703051
10 seconds input	0.670410	0.716058
ensemble (mean) 5 efficientnet_b0	0.686169	0.724319
ensemble (min) 5 efficientnet_b0	0.688977	0.734945
✅ ensemble (min) 6 efficientnet_b0	0.689146	0.738566
ensemble: mean[min(3 efficientnet_b0), min(3 regnety)]	0.691749	0.733836
✅ ensemble: mean[3 efficientnet_b0, 3 regnety]	0.690391	0.729178
Surprisingly, the results are very stable: correlation of public and private score is 0.96.

Inference Notebook: https://www.kaggle.com/code/chemrovkirill/birdclef-2024-1st-place-inference

Thanks for reading!

Project Details
Authors
Kirill Chemrov
chemrovkirill

Arseny Poyda
arsenypoyda


Share
36
14
11
17 Comments
2 appreciation comments
Hotness
Dieter
Posted a year ago

3rd in this Competition

Its not Psi’s CNN. I read this now several times in BirdCLEF solutions. Just because someone gets the chance to post submission summary doesn't mean he developed the solution alone. Its a team result.


Reply
Kirill Chemrov
Topic Author
Posted a year ago

1st in this Competition

You're right! Corrected


Reply
Muhammed Tausif
Posted a year ago

Exactly, correct.


Reply
Rinku Sahu
Posted a year ago

702nd in this Competition

Congratulations Team.
I'm trying to understand your approach, but I'm having difficulty with the "averaged labels" concept. Since each audio file has a single label, dividing it into 5-second chunks should still result in the same label for each chunk. Could you please clarify what you mean by "averaged labels"?

Model input
Models are trained on 10-second chunks that consist of two 5-second adjacent chunks with averaged labels. The idea is that a 10-second chunk provides processing of full chirps or full periods of chirps (if they are cropped by 5-second chunks).


Reply
Kirill Chemrov
Topic Author
Posted a year ago

1st in this Competition

Sorry for the late reply. You are right about train_audio data. As for unlabeled_soundscapes, we pseudo-label each 5-second chunk, so two adjacent chunks may have different labels.


Reply
Rinku Sahu
Posted a year ago

702nd in this Competition

That's great. Thank you for clarification 😀.


Reply
Cody_Null
Posted a year ago

293rd in this Competition

Great work, congratulations! Happy to learn from this one! I guess I should have spend more time with augmentations, pseudo labeling, and post processing!


Reply
Salman Ahmed
Posted 7 months ago

432nd in this Competition

Hi @chemrovkirill and @arsenypoyda, Thank you for sharing the detailed solution.

I have a question, when you created pseudo labels for unlabeled_soundscapes, how did you apply CE loss, if no specie was detected? Did you set a threshold to include or exclude those chunks?

If multiple species are detected in unlabeled_soundscapes, did you normalized it to have the sum of 1 for passing in to CE loss?


Reply
Arijit Pal
Posted 8 months ago

Sorry for the delayed comment, but I just had to say — this is truly impressive work.
Your solution is incredibly detailed, well-structured, and clearly thought through.
Really great job — it definitely deserves recognition!


Reply
thacrobatheskis
Posted 8 months ago

308th in this Competition

Sorry to comment so late after your post :) Thanks for the detailed description of the solution, as well as for the excellent presentation on youtube!

we calculated various statistics related to signal strength

I'm having trouble reproducing the same values that you got when calculating these statistics. By any chance were they computed after normalizing the samples using the global mean/std of all samples in the dataset?

Did you apply the same normalization to the audio samples before computing the mel spectrograms, or were the spectrograms derived directly from raw audio?


Reply
Sonu Jha
Posted a year ago

620th in this Competition

Can someone please tell me, how fold0.ckpt, fold1.ckpt, fold3.ckp has been generated ? I am not able to find the script in github.


Reply
Kirill Chemrov
Topic Author
Posted a year ago

1st in this Competition

We omitted script and config for training these models. However, the idea is simple. We just train the efficientnet_b0 model on the train_audio only. Folds 0, 1, 3 are chosen due to high public score.


Reply
Sonu Jha
Posted a year ago

620th in this Competition

Thanks, You explained it so well that it's easy for me to understand/replicate the top competition solution for the first time.


Reply
Metin Amedi
Posted a year ago

I appreciate the information you provided; it’s extremely helpful. Thank you!


Reply
Yassine Alouini
Posted a year ago

381st in this Competition

That's a great write-up, thanks for sharing and congratulations on the first position!


Reply
This comment has been deleted.

This comment has been deleted.


Appreciation (2)
CodeHacker
Posted a year ago

91st in this Competition

Nice work! Thanks for sharing.

Thibault Barthelet
Posted a year ago

854th in this Competition

thanks for this, very helpful !

Skip to
content
Kaggle

Create
Search

Home

Competitions

Datasets

Models

Benchmarks

Game Arena

Code

Discussions

Learn

More


View Active Events

Skip to
content
Kaggle

Sign In

Register
2nd place solution

BirdCLEF 2024

Solution Writeup · 2nd place · Jul 8, 2025

First of all, thanks to the organizers for this competition and congratulations to all winning teams!

Summary
Only first 5 seconds of each recording is used for training
EfficientNet B0 backbone
Performance boost by using pseudo labels from target domain
Ensemble of 6 models
Model diversity through different Mel parameters, data subsets, image sizes and probabilities to add pseudo labels
First experiments
The start in this year’s edition was not easy. Using models and training methods from last year didn’t work that well at first. Almost all attempts that improved local CV didn’t improve or even decreased public LB score and I was not able to get past 0.64 AUC for single models or 0.69 for ensembles.

Almost thinking about giving up, I restarted from scratch using the public notebook from @salmanahmedtamu following methods discussed by @lihaoweicvch. Thanks for sharing!

With this it was possible to get single model performance of 0.65/0.66 AUC.
The original notebook uses:

Only first 5s of training files and no extra files or classes from other sources
Model input: resized 3 channel Mel spec images of size 256x256
Backbone: eca_nfnet_l0
Mel parameters:
n_fft = 2048
hop_length = 512
n_mels = 128
f_min = 20
f_max = 16000
Training parameters:
CosineAnnealingLR scheduler with 5 warmup epochs
Peak learning rate 1e-4
100 epochs with early stopping if AUC is not improving for 7 epochs
Batch size 64
Average of BCE and FocalLoss
GEM pooling
Augmentations:
HorizontalFlip
CoarseDropout
Mixup of Mel spectrogram images within training batches
From this baseline I started to experiment with different backbones, hyperparameters, augmentation methods and image sizes. One major drawback of the model was its relative long submission time (> 1h). So besides improving the score, one objective was to reduce inference time to fit more models in an ensemble. For this I switched to an EfficientNet B0 backbone (tf_efficientnet_b0_ns) and reduced image sizes. It turned out, that results were pretty unstable regarding LB score (0.62-0.66) and sensitive to different combinations of Mel parameters and input sizes. But with further adjustments I was able to create models with inference time below 12 minutes, still keeping a score around 0.65 AUC.

Main changes to the original notebook include:

Backbone: tf_efficientnet_b0_ns
5 dropout layers before fc layer (inspired by BirdCLEF2023 4th place and BirdCLEF2021 2nd place)
Higher learning rate (1e-3), less warmup epochs (3) and less epochs (50)
Different Mel parameters (n_mels, hop_length)
Additional augmentation: local and global time/frequency stretching performed on Mel spec images via resizing parts and entire image
Creating checkpoint soup instead of using early stopping
Creating checkpoint soups follows the idea of model soups. But here, weights of the same model from different checkpoints from epochs 13-50 are averaged if they show an improvement in local CV score on one of the tracked metrics (LRAP, cMAP, F1, AUC). This led to more stable and sometimes even better LB scores.
With all above mentioned modifications, I was now able to create an ensemble of 6 models achieving 0.70 AUC on public LB. Not great but good enough to start playing around with pseudo labels created from the unlabeled dataset.

Performance boost with pseudo labels
Using pseudo labels from the test domain and their handling during training was the key element to move up to the top 10 in the leaderboard. Pseudo labels were created by applying the model ensemble with the best public LB score on the unlabeled data to create predictions for each 5s interval of each file.
After that, during next stage training, randomly selected 5s audio segments from the unlabeled test set are added to the training samples with a 25 % to 45 % chance. Before mixing the two audio signals, amplitudes of both waveforms are multiplied by a random factor.
The target vector of the training sample (1.0 at positions of primary and secondary species, rest all zeros) is combined with the pseudo label (vector with predicted probabilities) to form the new target vector by taking the maximum of both.

This method of using the pseudo labeled test data combines several advantages:

Noise augmentation: By mixing training samples with samples from the target domain the model learns how species sound within the environmental background noise of the test site habitat. This helps to address the domain shift between Xeno-Canto recordings and test soundscapes.
Additional training data: The model gets more training samples representing the noise characteristics and species distribution of the target domain.
Knowledge distillation: Since pseudo labels are derived from predictions of a stronger model (or ensemble of models in this case), its knowledge is transferred during training to the smaller model
After integrating pseudo labels, LB score significantly increased. The new ensemble was then used again to create a new set of pseudo labels. This cycle was repeated 3 times to iteratively improve model/ensemble performance. Progress on LB score (publ.|priv.) is shown in the following table:

Stage	Pseudo labels	Single model (ID 4)	Ensemble
0	-	0.65735 | 0.59270	0.70065 | 0.61738
1	From stage 0 ensemble	0.69165 | 0.66119	0.71090 | 0.67084
2	From stage 1 ensemble	0.69936 | 0.67445	0.72528 | 0.69035 *
3	From stage 2 ens. (normalized)	0.71154 | 0.67683	0.71716 | 0.69527


After the 2nd iteration pseudo labels needed some normalization (scaling to range 0…1 ) to allow stable model training because label minimum values got too large. Also, the stage 3 ensemble was not selected for final ranking because public LB score unfortunately didn’t show the expected improvement.
Performance and parameters of models from the 2nd place ensemble (2nd stage in previous table) are shown in the following table:

Params./Model ID	1	2	3	4	5	6
seed	42	42	42	42	70	42
n_folds	5	5	5	5	10	5
fold	4	1	4	4	0	4
dataset	bc24	bc24	bc24	bc24	bc24+	bc24
n_mels	128	128	128	64	64	64
hop_length	512	512	1024	1024	1024	1024
image_height	256	256	128	64	64	64
image_width	256	256	128	128	128	64
pseudoLabelChance	35 %	40 %	45 %	30 %	30 %	25 %
ampExpMin	-0.5	-1.0	-0.5	-0.5	-0.5	-0.5
ampExpMax	0.1	0.2	0.1	0.1	0.1	0.1
Inference time	~ 50 min.	~ 50 min.	~ 17 min.	~ 12 min.	~ 12 min.	~ 11 min.
Public LB score	0.73270	0.71975	0.71104	0.69936	0.69124	0.69309
Private LB score	0.68521	0.68533	0.68116	0.67445	0.64543	0.65862


Model diversity in the ensemble was realized by different Mel parameters, data subsets, image sizes, chances to add pseudo labels and amplitude factors to change the volume relation between training and pseudo label data. Parameters ampExpMin and ampExpMax provide the range for the random amplitude factor multiplied to training and pseudo label samples to change their volume in the mix: ampFactor = 10**(random.uniform(ampExpMin,ampExpMax))
Model 5 is the only one using external data. For this, additional files for the 182 species of the competition were downloaded from Xeno-Canto and the first 5 seconds part of each file was added to the training set (padded with zeros if too short).

Post-processing
Models were ensembled by simply taking the mean of predictions (probabilities from sigmoid outputs) of each single model. As a last step, for each file, predictions of a given window, were summed with those of the two neighboring windows with factor 0.5. This post-processing method was used by @theoviel and his team in the 3rd place solution of the Cornell Birdcall Identification competition .

Optimizations for inference
Parallel preprocessing of test audio files via multithreading
Precalculation of different versions of Mel spectrograms and reusing them for different models
Adding models using smaller image sizes as input
Setting a 2h timer to prevent submission timeouts for larger ensembles
What didn’t work
Too many things ;) Unfortunately, I found a good approach to use pseudo labels rather late in the competition. Some post deadline submissions show, that many things that didn’t work at the beginning are actually quite beneficial to improve private LB score if pseudo labels are included. Hope to continue on those experiments maybe at the next edition…

Inference Notebooks
All 6 models (Score may vary because submission time > 2h)
Best 2 models (Stable and better LB score with submission time ~1h10min)

Working Note
Lasseck M (2024) Improving Bird Recognition using Pseudo-Labeled Recordings from the Target Location. In: CEUR Workshop Proceedings.

Tagged
Computer Science
Environment
Neural Networks
Audio Classification
Project Details
Author
adsr
mariotsaberlin


Share
7
4
1 Comment
Hotness
Yassine Alouini
Posted a year ago

381st in this Competition

Well done on the second place and thanks for sharing your insights!


Reply
2
Skip to
content
Kaggle

Create
Search

Home

Competitions

Datasets

Models

Benchmarks

Game Arena

Code

Discussions

Learn

More


View Active Events

Skip to
content
Kaggle

Sign In

Register
3rd solution

BirdCLEF 2024

Solution Writeup · 3rd place · Jun 25, 2024

I am the one posting but this is team work with @christofhenkel and @theoviel : team NVBird!

Thanks everyone for the interesting competition, and congratulations to the winners ! We are very happy with the 3rd place, even though we missed the win by a very tiny margin.

Overview
Our pipeline is summarized below. A key ingredient of our solution is to use unlabeled soundscapes for pseudo labeling and model distillation. A number of models were trained with the training data, then used to predict labels on 5 second clips from unlabelled soundscapes. These were added to the original training data to train a new set of models used for the final submission.

Bird-CLEF-pipe

Data
Overall, we relied on knowledge acquired during previous competitions, and added some extra samples to fight class imbalance.

We use this year’s competition data plus the additional data from Xeno Canto shared in the forum, plus records from previous year competitions for the same species as this year. When a record name appeared in several competitions we picked the most recent one (their content is not always identical form one competition to the next one)
We capped the number of records per species to 500, keeping the most recent ones. Indeed, adding all extra data leads to severe class imbalance that was detrimental to model accuracy.
Low frequency classes are upsampled so that there are at least 10 samples for each class in the training folds.

To train models, we use the following preprocessing and augmentations:

We didn’t use all the training data. For each record we use a random crop of 5 seconds clip among the first 6 seconds, or the last 6 seconds of a record. If record length is smaller than 5 seconds, random padding so that the middle of the signal is between 2 and 3 seconds in the 5 sec resulting clip. For most of the models we used time shifting with a one second window as the only augmentation besides mixup. An exception are some models which are inspired by Birdclef23 2nd place SED models (https://github.com/LIHANG-HONG/birdclef2023-2nd-place-solution/blob/main/configs/sed_v2s.py)and use the same augmentations as used there
We use an additive mixup: primary labels are the max of primary labels of the two audios to be mixed. Secondary labels are the concatenation of secondary labels.
We mostly use image models that take log mel spectrograms as input. For these we compute mel spectrograms with parameters chosen to have an image size of 224x224 or 288x288 depending on the image model we use. Input waveforms are normalized to have a std of 1.

Models
First Level models
The cpu-only requirement was quite constraining for submissions, but this does not apply for pseudo-label generation, so we could use more backbones for first level models. When ensembling several models larger than the ones used at second level we perform what is known as model distillation. This is a rather powerful technique in general.

Models used include:

Efficientvit_b0.224.in1k on 224x224 log mel spectrograms
Efficientvit_b1.r288_in1k on 288x288 log mel spectrograms
A variety of CNNs (efficientnets, mobilenets, tinynets, mnasnets, mixnets) and Efficientvits (b0, b1, m3 trained on 224x224 log mel spectrograms.
SED model with tf_efficientnetv2_s_in21k on 128x313 log mel spectrograms?
We also fine-tuned aves-large and aves-base, which is a recent waveform based model.
Depending on the pipeline one or more of these models were used to predict pseudo labels on unlabelled soundscapes. We trained most models on full data with 5 different seeds.

Second Level models
We used efficientvit-b0 primarily and mnasnet-100 on 224x224 log mel spectrograms. Efficientvit-b0 showed great performances while still being very fast to infer. 5 folds take 40 minutes to submit using ONNX. We tried several models with similar throughput to effvit-b0, and decided to also use an mnasnet-100 for diversity. Since computing log mel spectrograms is slow on CPU we decided to use the same mel spectrogram hyperparameters for all models in the inference notebook, to only have to do log mel spectrogram transformation once and use the same input for our ensemble models.

For training second level models we added the unlabelled soundscapes with the predicted pseudo labels to the training data. This looks simple but it took several attempts to find the correct way to do it. What worked fine was to use rather large batch sizes (128),
We used the two following strategies;

Add extra soundscapes to each batch. Each soundscape is split into 5 second clips, which means we added 48 x 4 = 192 clips with pseudo labels to the 128 samples with actual labels in each batch.
Add 128 samples with pseudo labels, taken from random soundscapes this time.
Final ensemble
At the end we had an ensemble of 14 model weights via 3 pipelines, one pipeline per team member:

Pipeline 1 - 5 seeds
Level 0: efficientvit_b0
Level 1: efficientvit_b0
Pipeline 2 - 2 seeds + 2
Level 0: efficientvit_b1, mobilenetv2, efficientnet_b0, efficientnetv2_b0, efficientvit_b0, efficientvit_m3, aves-base, aves-large

Level 1: 2x mnasnet-100

Level 0: Efficientvit_b0, mixnet_s, mnasnet_100, tinynet_b, efficientvit_b0, efficientvit_b1, mobilenetv3

Level 1: 2x efficientvit_b0,

Pipeline 3 - 5 seeds
Level 0 efficientvit_b1_288, 5x efficientnet_v2_s sed, aves-base, aves-large
Level 1 5x efficientvit_b0
That ensemble scored 0.689970 private, and 0.742124 public.

Training
It was a bit tricky to tweak parameters without a validation set. We have two training pipelines with different parameters, and it is rather unclear what actually mattered.

We used BCEWithLogitsLoss, without any label smoothing. Labels were defined by primary labels. Secondary labels were used to mask loss: loss for secondary labels is multiplied by 0. Reason is that we don’t know if a secondary label occurs at the start and at the end of the record, but they could. Given the uncertainty, we mask the loss for these. Masking secondary label loss improves LB by about 0.01.

When using pseudo labels on unlabelled soundscapes we set their secondary labels to be empty.

We used AdamW or Ranger optimizers. The number of epochs for the second level was way higher than for first level ones. For instance in one pipeline first level models are trained with 30 epochs while second level models are trained with 88 epochs.

Post Processing
We used several post processing, to incorporate soundscape-level information.

The first one worked well on public LB with a 0.02 boost initially. Experiments after competition end show that its effect is much smaller (0.004 on our best unselected submission), and even detrimental on our best selected submission (- 0.002). The idea is that if a bird appears anywhere in the soundscape then its probability to appear in any 5 second slice is increased.
For a given soundscape, once we have a 48x182 logits array or predictions P, we compute the maximum P_max of P over the time dimension. We then replace P with:
P + (P_max + P.mean() - P_max.mean()) * 0.8.
The 0.8 weight can be tuned further.

The second postprocessing had a smaller impact on LB the first time it was tried but it had a larger impact of +0.01 on our best selected sub. The idea is to smooth predictions for each 5 second clip by blending it a bit with the previous two and the following two clips. We used a convolution for smoothing predictions with the kernel [0.1, 0.2, 0.4, 0.2, 0.1].

What did not work
A lot. Main issue was that we could not find a reliable local validation scheme. We mainly looked at train data cross-validation like most of the participants, but also put some effort in looking at past BirdCLEF train / test differences individually for each year. We thought that this might help to understand what augmentations might be bridging the train (= xeno-canto) / test (= PAM soundscape recordings) gap. However, the results were quite inconclusive. Probably because soundscapes are quite different each year. Whatever we tried lost correlation with the public LB once scores were high enough.

One thing that seemed to work great and didn’t in the end was the post processing based on max probability per soundscape.

Another one was the Aves model which improved public LB by about 0.01 each time it was used, but appeared to be detrimental on private LB.

We tried a number of augmentations, including those documented in previous competitions or in academic papers, but none seemed to really help.

ONNX did not bring much speedup (maybe 10%) and Openvino was much faster (almost 2x speedup compared to pytorch). However we noticed a drop of about 0.01 when using Openvino compared to ONNX, and decided not to use it. Maybe the speedup from openvino would have been compensated by the larger number of models we could use in the final ensemble?

Our final model selection did not work that well in the end. We had individual models submitted with private scores above 0.70, but we did not include them in our ensemble given their relatively low public scores.

Thanks for reading !

Edit, link to code (alphabetical order):

CPMP's part
Dieter's part
Theo's part
Best selected submission: https://www.kaggle.com/cpmpml/birdclef-2024-inf-ens-08
Project Details
Authors
CPMP
cpmpml

Dieter
christofhenkel

Theo Viel
theoviel


Share
1
19
16 Comments
Hotness
Shubham Thapa
Posted a year ago

402nd in this Competition

Congrats for the gold and thankyou for the writeup !

I have some questions aswell :-

1) Regarding loss ( Secondary labels were used to mask loss ) that means you were not giving any importance to secondary lables at all ?

2) Regarding Full data training ( We trained most models on full data ) how did you made sure that model is not overfitting ? i did the same but model got overfitted i guess ( i added lots of augs / background noise etc , tried to make my data hard to learn on classes ) which were abundant and also oversampled my other classes with some techniques .

3) Regarding PsudoLabels ( what was your threshold to determine psudo labels and there can be multiple classes in the audio clip or the time frame which you were using to infer so i am assuming you might be using 1 label which has the max conf ? I did the same but i guess my model was overfitting.

4) Regarding Upsampling ( did you just made multiple copies of the audio clips or added some transformation on those ) ? I upsampled the classes by adding some augs to the original 5 sec clip and also added unlaballed soundscapes to some other copies , ( i took only those soundscape clips where my model was getting very low confidence like < 0.05 ) but didnt got any luck probably becuase at the end i overfitted my model .

Again Thankyou for your time and effort .


Reply
CPMP
Topic Author
Posted a year ago

3rd in this Competition

Congrats for the gold and thankyou for the writeup !

Thank you on behalf of the team.

that means you were not giving any importance to secondary lables at all ?

I think yes (not sure about the meaning of "importance"). We do find secondary labels to be important as we use them.

how did you made sure that model is not overfitting ?

We didn't really. We did train models using 5 fold CV, but the CV score wasn't correlated with LB score once we reached 0.65 or above. on LB. I welcome any hint on how to properly evaluate models in this competition besides submitting them.

Regarding PsudoLabels ( what was your threshold to determine psudo label

We use the predicted probability as label. We don't round probabilities to binary labels.

i am assuming you might be using 1 label which has the max conf ?

We don't. The competition is a mult label competition, not a multi class competition.

Regarding Upsampling ( did you just made multiple copies of the audio clips or added some transformation on those ) ?

We duplicated the record, then augmentations (time shift and mixup) are applied independently on each copy.


Reply
Shubham Thapa
Posted a year ago

402nd in this Competition

Thankyou for the answers .

We don't. The competition is a mult label competition, not a multi class competition.

What i meant by that was how you were deciding the primary label for your psudo labels in the window of your time frame and i guess it will be your max confidence class ? since you are setting your secondary classes as empty as mentioned in your write up.


Reply

3 more replies
Profile picture for CPMP
Profile picture for Shubham Thapa
Shiro
Posted a year ago

14th in this Competition

Congratulation for the gold and thanks you for the write up !

I have few questions:

(1) regarding the pseudo labels approach, does smaller batch was also giving you positive results? I tried 16 but did not get any luck unfortunately … That was just providing more stable LB.
(2) how did you pick which checkpoint to select for inference?

Reply
CPMP
Topic Author
Posted a year ago

3rd in this Competition

Congratulation for the gold and thanks you for the write up !

Thank you on behalf of the team.

does smaller batch was also giving you positive results?

We tried as large batch as possible. 256 seemed to lower performance hence we stopped at 128. We then found that the more pseudo labels per batch the better. Maybe we could have gone further.

how did you pick which checkpoint to select for inference?

The last epoch one. We don't use early stopping.


Reply
1
Aaditya Porwal
Posted a year ago

25th in this Competition

Congrats for the gold and 3rd position and thankyou for the writeup !!!


Reply
Phaedrus
Posted a year ago

297th in this Competition

Congratulations and great writeup.

Could you clarify a little about masking loss for secondary labels? From what I can understand, if a bird X is present as secondary label, you essentially mask loss for this bird/index in BCEWithLogitsLoss? Is this the correct interpretation?

Also, will you be opening up your inference sub for us to look at optimizations and stuff? 😐


Reply
CPMP
Topic Author
Posted a year ago

3rd in this Competition

Congratulations and great writeup.

Thanks on behalf of the team.

if a bird X is present as secondary label, you essentially mask loss for this bird/index in BCEWithLogitsLoss?

Yes, we mask as you say.

I documented this idea in my 2020 Birdclef writeup. I reproduce it here:

Primary labels were noisy, but secondary labels were even noisier. As a result we masked the loss for secondary labels as we didn't want to force the model to learn a presence or an absence when we don't know. We therefore defined a secondary mask that nullifies the BCE loss for secondary labels. For instance, assuming only 3 ebird_code b0, b1, and b2, and a clip with primary label b0 and secondary label b1, then these two target values are possible:

[1, 0, 0]

[1, 1, 0]

The secondary mask is therefore:

[1, 0, 1]

I hope this is clearer now.

will you be opening up your inference sub

I'll check with my team members.


Reply
1
Phaedrus
Posted a year ago

297th in this Competition

Thankyou!


Reply
Cody_Null
Posted a year ago

293rd in this Competition

Wow great work! I may have to go back to the board a bit. I had used efficientvit_b0 and it took 25 min to run on ONNX. 5 in 40 minutes is crazy, I couldnt get any of my models to be that fast! I would love to hear more about how that worked


Reply
CPMP
Topic Author
Posted a year ago

3rd in this Competition

Wow great work!

Thank you on behalf of the team.

I would love to hear more about how that worked

We didn't do anything fancy there. We used the model.half().float() trick that was shared, this shaved few minutes per 5 checkpoints. But we use various tricks like loading records once for all models, computing mel spectrograms once for all models, etc.


Reply
Dieter
Posted a year ago

3rd in this Competition

melspec transformation takes quite some time. But even we had 5 models we only need to do it once. So you can check how long multiple effnet b0 of you would take, if you only do melspec once


Reply
Hugo de Heer
Posted a year ago

247th in this Competition

Congratulations with gold first of all and thanks for your clear writeup!

How did you manage to submit 5 fold efficientnet_b0 224x224 within 40 minutes? For us, without ONNX it took around ~1h for a single fold, and with ONNX we observed a 2x speedup to around ~30 minutes per fold. Still around ~3.8x slower than your approach.

How did you use the different seeds? Did you just pick the seed with highest public LB for submission?

How did you exactly merge your pipelines, I am a bit confused by your ensembling section. Did you both submit level 0 and level 1, and which models were included?

Cool to see that you also tried the convolving postprocessing smoothing, we observed that surprisingly smoothing the current 5 seconds with the mean of the whole 4 minutes was even better than the convolved kernel, wonder if you also tried that.

Thanks!


Reply
CPMP
Topic Author
Posted a year ago

3rd in this Competition

Congratulations with gold first of all and thanks for your clear writeup!

Thank you on behalf of the team.

How did you manage to submit 5 fold efficientnet_b0 224x224 within 40 minutes?

Efficientvit_b0, not efficientnet_b0 :D

How did you use the different seeds? Did you just pick the seed with highest public LB for submission?

Certainly not. This is a recipe for overfitting. We either used random seeds, or fixed seeds we change from time to time (team mates have differnent views here, but we all agree that tuning the seed is a disaster)

How did you exactly merge your pipelines,

We took the average of the logits of the 14 models.

smoothing the current 5 seconds with the mean of the whole 4 minutes was even better than the convolved kernel,

We tried and found that smoothing with the max over the soundscape is even better. Till we saw that it isn't with high score on private LB.


Reply
Hugo de Heer
Posted a year ago

247th in this Competition

Ah great! Thanks for the clarification!

And then I assume that you submitted all Level 1 models from each pipeline together, and then you used a varied ensemble of bigger models for Level 0 to get better pseudo labelling for Level 1? That is a very nice idea, we tried pseudo-labelling with the Google model on unlabelled soundscapes. That did not work out for us, unfortunately. Your approach on training multiple larger models on Xeno-Canto and using that for pseudolabelling seems to be a better approach :)


Reply
CPMP
Topic Author
Posted a year ago

3rd in this Competition

Actually the best second level model we selected was using pseudo labels from effcientvit_b0 only!

The other two pipelines used Aves and looked great on public LB, but were not great on private LB unfortunately.

Best unselected pipelines were similar to those in the ensemble but without Aves.


Reply
CPMP
Topic Author
Posted a year ago

3rd in this Competition

We shared our code, link in the post.


Reply
Anirudh Mittal
Posted a year ago

Thanks for the insight! Can you also share the code?


Reply
Metin Amedi
Posted a year ago

Congratulations 🎊


Reply
Muhammad Azeem
Posted a year ago

Congratulations and nice writeup.


Reply
Skip to
content
Kaggle

Create
Search

Home

Competitions

Datasets

Models

Benchmarks

Game Arena

Code

Discussions

Learn

More


View Active Events

Skip to
content
Kaggle

Sign In

Register
4th place solution: Team Cerberus

BirdCLEF 2024

Solution Writeup · 4th place · Jun 23, 2024

Thank you to Kaggle, the hosts, and all the competitors. Participating in this exciting competition has been an amazing experience. Here's a look at our 4th-place solution. This achievement was truly a team effort, with equal contributions from @ajobseeker and @tamotamo. I'm grateful to have had the chance to work with them in this competition.

Update (2024-06-23):
Added the inference notebook and training code.
Inference Notebook: https://www.kaggle.com/code/yokuyama/bc24-4th-place/notebook
Train code(melspec models): https://github.com/yoku001/BirdCLEF2024-4th-place-solution-melspec
Train code(raw signam models): https://github.com/tamotamo17/BirdCLEF2024-4th-place-solution-raw-signal

TL;DR
Ensemble of Melspec Models and Signal Models
TTA
OpenVINO
Post Processing
Scores
Model	Public Score	Private Score	Public Score (+TTA)	Private Score (+TTA)
Melspec Model B (inception-next-nano)	0.668	0.623		
Melspec Model A (rexnet_150)	0.676	0.641	0.690	0.649
Melspec Model A (seresnext26ts)	0.682	0.645	0.693	0.651
Raw signal Model C (tf_efficientnet_b0_ns)	0.673	0.620	0.691	0.636
Weighted Mean	0.717	0.667	0.731	0.676
Weighted Mean + Geometric Mean			0.732	0.677
Weighted Mean + Geometric Mean + Smoothing			0.741	0.685
Weighted Mean + Geometric Mean + Smoothing + Cut-off (Final Sub)			0.7469	0.6877
Weighted Mean + Geometric Mean + Smoothing + Cut-off + Max With Neighbors			0.749	0.689
Weighted Mean
0.15*Model B + 0.25*Model A (rexnet_150) + 0.3*Model A (seresnext26ts) + 0.3*Model C
Geometric Mean
(0.15*Model B + 0.25*Model A (rexnet_150) + 0.3*Model A (seresnext26ts) + 0.3*Model C) + 0.3*(Model A (rexnet_150) * Model C)**(0.5)
By adding the geometric mean of the MelSpec Model(Model A) and the Raw-signal Model(Model C) to the ensemble, we achieved a slight improvement in our score. Choosing this ensemble at the end allowed us to fortunately remain in the prize-winning positions.
We saw a big risk of overfitting, so we decided not to spend any more time adjusting the ensemble weights.
Model A: 2021-2nd Melspec CNNs
We heavily referenced the code from the 2023 2nd place solution to build our training and inference pipeline for this model. Big thanks to @honglihang for sharing such valuable information. Their codebase was incredibly strong, and with just a few modifications, we were able to create a single model with an LB score of 0.68.

Dataset

BC2024
Some models pretrained on 2021, 2022, and 2023's datasets.
Random 15-20 seconds from audio at training, first 5 seconds at validation.
Preprocessing

n_mels=128, n_fft=2048, f_min=0, f_max=16000, hop_length=627, top_db=80.
Data Augmentation

AddBackgroundNoise (datasets)
Gain
Noise Injection
Gaussian Noise
Pink Noise
Mixup
Model

seresnext26ts
rexnet_150
2021-2023 pretrained
Loss Function

BCELoss
Class sampling weights proposed by 1st place of 2023 competition.
Model B: Simple Melspec CNNs
Dataset

BC2024 + xeno-canto-additional-cleaned(see Validation Strategy section)
Random 5 seconds from audio for training, first 5 seconds for validation.
Data Augmentation

AddBackgroundNoise (datasets)
Gain
Noise Injection
Gaussian Noise
Pink Noise
Mixup
Sumup
Model

inception-next-nano with attention head

InceptionNeXt with the same scaling as ConvNeXt-nano
from timm.models.inception_next import _create_inception_next
from timm.models.inception_next import InceptionDWConv2d
from timm.models._registry import register_model

@register_model
def inception_next_nano(pretrained=False, **kwargs):
    print("inception_next_nano")
    model_args = dict(
        depths=(2, 2, 8, 2), dims=(80, 160, 320, 640),
        token_mixers=InceptionDWConv2d,
    )
    return _create_inception_next('inception_next_nano', pretrained=False, **dict(model_args, **kwargs))
Model C: Raw signal CNN
This model is inspired by the HMS 2nd place solution. A big thanks to @cooolz! They provided a detailed explanation of them solution.

Dataset

BC2024
Removed duplicate data by referring to [this link].
Added several samples from the minority class using data from xeno-canto.
Applied stratified 5-fold cross-validation grouped by author.
Classes with fewer than 15 samples were upsampled to 15 samples during training.
Preprocessing

Used the first 5 seconds of each audio sample.
Downsampled the audio to half the original rate (from 32000 Hz to 16000 Hz).
Reshaped the downsampled audio data from a size of 80000 to 625x128.
Data Augmentation

Annotated 50 background segments from unlabeled data and added them as background noise.
Gain
Noise Injection
Gaussian Noise
Pink Noise
Random Volume
Mixup
Cutmix
Model

tf_efficientnet_b0_ns with SED head
Loss Function

focal loss
Validation Strategy
Using the training data for validation didn't give us reliable results, so we switched to a synthetic data approach.

We sampled files for 40 out of 182 classes from the xeno-canto-additional dataset and cropped the segments where the birds were vocalizing to create a clean dataset.
We sampled audio files containing only background noise (without bird calls) from the unlabeled soundscape dataset.
We combined the clean dataset and background noise to create a test-like dataset with time-series labels.
Using this synthetic dataset, we calculated the ROC AUC score.
Although this validation method did not perfectly correlate with the LB results, it provided more reasonable outcomes compared to using the first 5-second crop method. For more details, please refer to this notebook.

TTA
To enhance the accuracy of our time series predictions, we employed techniques similar to sub-pixel super-resolution. Instead of predicting just the 5-second frames during inference, we also predicted frames shifted by 2.5 seconds. We then combined these results as a TTA. This method helped in refining the overall predictions.

img

OpenVINO + INT8 Post Training Quantization
To speed up our model's inference time, we used OpenVINO. Additionally, we implemented post-training quantization to convert our model to INT8.

For the quantization calibration dataset, we used our model's training dataset, applying augmentations like background noise addition and gain changes. We believed that these augmentations would help create a quantized model better suited to handle a wider range of test data scenarios.

The results were impressive: our inference speed improved dramatically, with the quantized model running 30-40% faster.

When performing quantization, the selection of layers to be quantized was crucial. We observed that excluding the head layers from quantization tended to improve the model's accuracy.

names = ['/head/Gemm/WithoutBiases', '/global_pool/Pow', '/global_pool/GlobalAveragePool', '/global_pool/Pow_1', '/global_pool/Clip']
quantized_model = nncf.quantize(
    model, calibration_dataset, subset_size=600,
    ignored_scope=nncf.IgnoredScope(names=names),
)
We began working on quantization just three days before the submission deadline, leaving us insufficient time to thoroughly verify the combination of ensemble and quantization. So, we used a quantized model in only one of our two final submissions. (trade-off: we reduced the number of TTA runs for this submission.)

We ended up finding that enabling quantization did not significantly impact the public/private scores.

Post Processing
By using several tricks, we were able to improve our scores by approximately 0.01 on both the private and public leaderboards.

Smoothing
Similar to the 6th place team, we improved our scores by taking the moving average of adjacent segments.
Cut-off
Birds that appear once in 4 minutes of audio are more likely to reappear compared to other audio. Recognizing that the probability could be low due to noise, overlapping calls with other birds, and inference slices cut by bin boundaries, we halved the value if the model had a confidence of 0.10 or less in all 48 bins of the 4-minute audio. In other words, we halved the probability if there were no birdsong (0.10 or less) in all 48 sections, and left the number unchanged if there was birdsong at least once in any of the 48 sections.
Max With Neighbors
Select the maximum value including the previous and next two rows. For a 30-second test sample, if the inference values of a label are [0.1, 0.3, 0.5, 0.2, 0.4, 0.1], modify them to [0.5, 0.5, 0.5, 0.5, 0.4]. We conducted many tests with different datasets and found a 25% probability of score decrease, so it was not included in the final submission.
What didn't work
Pseudo labeling using unlabeled data.
Data cleansing and hand-labeling for training data.
Using novel loss functions. BCE and Focal Loss performed almost the best.
Manifold Mixup, D-Mixup
PCEN
CWT, CQT, VQT
Trainable frontends: Leaf, trainable filterbank, trainable stft, Conv1D
Reparameterized model
Mobilenet V4
BirdNET embeddings
Project Details
Authors
HB
ajobseeker

yokuyama
yokuyama

TmT
tamotamo


Share
6 Comments
Hotness
HB
Posted a year ago

4th in this Competition

First of all, I would like to thank Kaggle and the hosts. And I am especially grateful to honglihang for providing the 2023 2nd place solution. His solution was remarkably perfect and was both a great help and a valuable learning experience. yokuyama and tamotamo were the best teammates, and I had an incredibly perfect time with them. This was my first gold medal, and I am deeply thankful for the unforgettable experience we shared.


Reply
CPMP
Posted a year ago

3rd in this Competition

Congrats.

I sue the same TTA as you in a previous comp, and I tried it here early. The gain in LB score looked small for a double in inference time hence we did not pursue it. Maybe we should have given how good it is for you.


Reply
yokuyama
Topic Author
Posted a year ago

4th in this Competition

Thanks. Congrats to you too!

I am honored to have come up with the same idea as a GM like you. While our team was limited in the number of models we could use for this TTA, we believe it was a worthwhile trade-off.


Reply
Arjun M97
Posted a year ago

755th in this Competition

Congratulations on your achievement. I had a question about your approach how do you get these weights for the ensemble models, do you use the loss function to get this weight(importance of their press) for these models or is it based on how each model performs and then decides the weightage for the model.


Reply
yokuyama
Topic Author
Posted a year ago

4th in this Competition

Thanks.

For our ensemble weights, we primarily relied on public LB scores and validation scores using synthetic-data to make rough estimations.

We didn’t focus too much on the performance of individual models when deciding these weights. Given that we didn't manage to develop any standout models in this competition, we concentrated on exploring combinations that might boost our overall score through ensemble, even if some individual models had lower scores.


Reply
1
Mubashir Ahmed Siddiqui
Posted a year ago

congratulations @ajobseeker @yokuyama !


Reply
Skip to
content
Kaggle

Create
Search

Home

Competitions

Datasets

Models

Benchmarks

Game Arena

Code

Discussions

Learn

More


View Active Events

Skip to
content
Kaggle

Sign In

Register
5th solution

BirdCLEF 2024

Solution Writeup · 5th place · Jun 13, 2024

First of all, thanks to the organizers for hosting this competition.
And congrats to all winners.

Data
2024 data

Model
model	backbone	weights	public	private
raw signale	hgnetb0	3/5 folds	0.720354	0.666671
spectrum	efficientnetb0	1/5 folds	0.708198	0.672360
raw+spectrum	hgnetb0+effb0	1/5 folds	0.698408	0.660503
ensemble	…	0.5+0.4+0.1	0.743960	0.687173
1. raw signal model
# x bsx160000
#bsx80000x2
x = x.view(bs, -1, 2)           
#bsx2x80000
x = torch.transpose(x, 2, 1)    
#bsx2x1000x80
x = x.view(bs, 2, -1, 80)         
feature = self.backbone(x)   
2. spectrum model
torchaudio.transforms.MelSpectrogram(
              32000,
              n_mels=512,
              f_min=0,
              f_max=16000,
              n_fft=2048*2,
              hop_length=512,
              normalized=True,
          ),
torchaudio.transforms.AmplitudeToDB(top_db=80)
3. mix model
raw_f  = self.raw_model(x)
spec_f = self.spec_model(x)
feature = torch.cat([raw_f,spec_f],dim=1)
x=self.fc(feature)
## spectrum is resized to 256x256 for fast speed, 
## other params for mel spectrum is the same with spectrm model.
Preprocess
1. wave/max(wave) if max(wave)>1
Augmentation
XY cut out for spectrum model
Train
1. Random sample 5 seconds data for trainning,
first 5 seconds data for validation.
2. Mixup p=1.
3. BCEWithLogitsLoss
inference
openvino
extra
I make a post to discussion the method feeding raw signal to vision model. Here is the link good luck
inference notebook

Project Details
Author
coolz
cooolz


Share
1
33 Comments
Hotness
Donghui Zhang
Posted a year ago

11th in this Competition

Regarding the "raw model", how did you deal with it to achieve good results? I rarely see people using raw signals in Bird competitions.


Reply
coolz
Topic Author
Posted a year ago

5th in this Competition

I add some comments one the pse codes, i think you can get it now.


Reply
CPMP
Posted a year ago

3rd in this Competition

Good work. I was betting you'd use packed raw waveform as you did in HMS competition!

Congrats on becoming a GM. It was a matter of time given what you did in HMS.

Did you compare openvino accuracy with pytorch models? Asking because we found that it loses some accuracy and we didn't use it as a result.


Reply
coolz
Topic Author
Posted a year ago

5th in this Competition

Thanks.

I noticed that you'd mentioned the precision loss with openvino.
I manully check some of the outputs of pytorch, onnxruntime, and openvino, and notice very small differences, but didn't try to sub with onnxruntime. By the way, as the the score is rank relavent, i use logits as final output.


Reply
CPMP
Posted a year ago

3rd in this Competition

I used logits too until we found that smoothing worked better when applied to probabilities.


Reply
coolz
Topic Author
Posted a year ago

5th in this Competition

Lesson learned, i should have checked previous competition carefully. I saw several solutions that use postprocess that worked. I didn't use post process and previous data, a little bit pity.


Reply
Dieter
Posted a year ago

3rd in this Competition

Congrats for reaching Competition Grandmaster status 🎉 Huge achievement


Reply
2
Shiro
Posted a year ago

14th in this Competition

Congrats for your medal!

Question about your signal approach, what is the intuition behind your reshaping? Why picking 2 channels instead of 3 for example?


Reply
coolz
Topic Author
Posted a year ago

5th in this Competition

I want to make a 1 channel input ,but the input size is too big.
And that makes it too hard to do ensemble.
Then i reshape it as 2 channels( [123456]->[[135], [246]]). In some way, you can regard the input as 2 waves that with 16000hz sample rate.


Reply
Bartley
Posted a year ago

342nd in this Competition

Nice use of 1D-signals again! Also, congrats on becoming GM.

Is there a reason why you use n_mels=512 and then resize to (256, 256) rather than just setting n_mels=256?


Reply
coolz
Topic Author
Posted a year ago

5th in this Competition

For the spectrum model i use raw spectrum, but for the mix model, i use resized 256x256 for inference speed reason, otherwise, it would be timeout.


Reply
Bartley
Posted a year ago

342nd in this Competition

Hmm okay, thanks for the reply.

Why you did you not set n_mels=256? This should make the height match the resize shape?

eg.

import torch
import torch.nn as nn
import torchaudio

l= nn.Sequential(
    torchaudio.transforms.MelSpectrogram(
              32000,
              n_mels=256, # <- change this to 256?
              f_min=0,
              f_max=16000,
              n_fft=2048*2,
              hop_length=512,
              normalized=True,
          ),
torchaudio.transforms.AmplitudeToDB(top_db=80),
)
x= torch.rand(32_000*5)
print(l(x).shape)

# resize here..

Reply
coolz
Topic Author
Posted a year ago

5th in this Competition

Yeah， My models need raw data, spectrum512, spectrum256.
Therefor, the dataiter produce raw, spectrum512, spectrum256(resized by spectrum512). In this way i just need to load every ogg file one time to reduce the time cost.
I use n_mel=512 because i developed my spectrum model( input size 512,313) early and that is the best params for my model.


Reply
slime
Posted a year ago

449th in this Competition

Great how you managed to train raw model that well, in my experiments it had bad CV so I didn't submit it


Reply
Ayush Solanki
Posted a year ago

928th in this Competition

Hi, I had a doubt like taking 5 random seconds from sample during training wont guarantee the bird noise in it right? If yes then how the model works?


Reply
coolz
Topic Author
Posted a year ago

5th in this Competition

I see some people train with first 5 sec data. But in my experiments, random sample 5 sec data, perform better.


Reply
snehal
Posted a year ago

286th in this Competition

Solution seems so simple! How did you get such powerful single models using this approach? Were there any other tricks you used or parameters you optimized to get such high scoring models? Im so curious how such basic settings score so high. And congrats on GM rank!


Reply
snehal
Posted a year ago

286th in this Competition

Would you be willing to share training code?


Reply
coolz
Topic Author
Posted a year ago

5th in this Competition

Thanks.
The model is quiet simple, but it's sensitive to random seed. I use lb to check model performance. Single model is not that powerful. But the ensemble of raw signal and spectrum helps a lot.


Reply
snehal
Posted a year ago

286th in this Competition

What I’m asking is how did you get the spectrum only model to 0.70 public score when the baseline score given by most public notebook or discussion showed only 0.66 public score for spectrum only.


Reply
RihanPiggy
Posted a year ago

7th in this Competition

Impressive!! Will you mind providing some more details about the raw signals model?
I think it is really a big breakthrough because current solutions like birdnet, bird-vocalization-classifier are all based on computer vision, thus solution with raw signal maybe cutting edge.


Reply
coolz
Topic Author
Posted a year ago

5th in this Competition

Like the code below,

x = x.view(bs, -1, 2)
x = torch.transpose(x, 2, 1)
x = x.view(bs, 2, -1, 80)
then feeding to a 2d-CNN

And i add some comment one the pse codes, i think you can get it now.


Reply
Shubham Thapa
Posted a year ago

402nd in this Competition

I am not sure if it makes sense, but here is my thought since the model was trained on raw waves model might had learned to predict very well for the noisy data , it might have created some range pass that range it might be treating those features as an outlier, since the test data had too much noise it would had discarded all those unnecssary features upto some range based on training data, and since we normalize the value for mels we might not had those outliers , i am still confused tho :)


Reply
Muhammad Azeem
Posted a year ago

Congratulations 👏


Reply
Aaditya Porwal
Posted a year ago

25th in this Competition

Congrats for becoming Competition Grandmaster!!!!!


Reply
faithk7u
Posted a year ago

668th in this Competition

Congrats for becoming a competition GM! Huge achievement! Also impressive to see the relatively simple solution can achieve the great final result.

Have the following questions:

Why did you choose 5-sec clip during the training stage? In my experience, 10-sec clip from the middle of the train audio improves the cv score a bit, and also gives the model more samples to learn from my opinion.

Why didn't you use more image/audio augmentations?

Is there a reason for choosing n_fft=2048 * 2? That seems a lot more computation required than when n_fft=512, which some solutions adopt to use.

Thanks again for sharing the solution!


Reply
Yassine Alouini
Posted a year ago

381st in this Competition

Well done. The idea to mix raw and MEL spectrogram is great.
Could you give more details about the openvino part, thanks!


Reply
coolz
Topic Author
Posted a year ago

5th in this Competition

Like most people, convert pytorch model to onnx, then use openvino do inference. But spectrum is produced by torchaudio. And I will share my notebook later.


Reply
2
YoungJun_Player
Posted a year ago

Congratulations! I've just started studying I feel that I can grow quickly with the help of people like you, who are willing to discuss, share, and provide guidance. Thank you.


Reply
C R Suthikshn Kumar
Posted a year ago

27th in this Competition

Congratulations on securing 5th Place in this competition. Also, thanks on sharing the solution details.


Reply
vialactea
Posted a year ago

12th in this Competition

Thanks for sharing and congratulations on the 5th place and GM title!

I'm surprised that 1D models worked. We tried them early and discarded the idea because the performance was considerably worse than melspecs. Maybe the size of the test made a difference (only part of the data for 1 fold and 10 epochs). A reminder to revisit early decisions.

Could you please elaborate a bit on the approach you followed to evaluate models and define the weights for the ensemble? This was a struggle for us and I wonder how other teams approached it.


Reply
coolz
Topic Author
Posted a year ago

5th in this Competition

I didn't find a good way to do local cv, and raw signal model cv is lower than spectrum model.
I use 0.5,0.4,0.1 as ensemble weights. Because raw signal model score is about the same with spectrum model, and the mix model is a combine of them, use a small weight here.


Reply
Phaedrus
Posted a year ago

297th in this Competition

Congratulations ! Would you be publishing your inference pipeline? I'd be very interested to see how you optimised MelSpec layers with openvino, I failed at this.


Reply
coolz
Topic Author
Posted a year ago

5th in this Competition

I will share it later. However I met the same problem, I use torchaudio to produce the spectrum in dataiter, the layer not included in the model.


Reply
Aphysict
Posted a year ago

9th in this Competition

Try my last year notebook


Reply
Phaedrus
Posted a year ago

297th in this Competition

Thanks, I will go through your NB.


Reply
This comment has been deleted.

This comment has been deleted.

This comment has been deleted.

Skip to
content
Kaggle

Create
Search

Home

Competitions

Datasets

Models

Benchmarks

Game Arena

Code

Discussions

Learn

More


View Active Events

Skip to
content
Kaggle

Sign In

Register
6th Place Solution

BirdCLEF 2024

Solution Writeup · 6th place · Jun 12, 2024

First, I would like to thank the hosts for organizing such a very interesting competition and the kaggle team for facilitating it so smoothly. Also, congratulations to all the top winners.

My solution is based on the last year's 2nd place solution by @honglihang. I would like to thank him greatly. The difference between his last year's solution and mine is mainly pseudo-label and post-processing.

CV Strategy
CV was not helpful for me, so I tuned my models by checking public LB. To avoid overfitting to public LB, I reduced submissions and gave up tuning minor hyperparams.

One of the final submissions is my public best model. Using additional training data made public LB about 0.02 worse, but this was counter-intuitive and I expected these to be reversed in private LB, so I selected it for the other one.



Pseudo Labeling
model.forward() method input several pieces of 5 seconds together, but calculate the loss without aggregation when using pseudo-labeled data, and after aggregation with max for competition data. Hard labeling by threshold did not improve scores.

This improved by 0.02~0.03 in publc and 0.015~0.04 in private before ensemble.

""" model.forward() """
total_loss = 0
for i in range(bs):
    start = i * self.factor
    end = (i + 1) * self.factor

    if is_pseudo[i] == 0:
        this_logits = torch.max(logits[start:end], dim=0, keepdim=True).values
        this_y = torch.max(y[i], dim=0, keepdim=True).values
        this_weight = torch.max(weight[i], dim=0, keepdim=True).values

        loss = self.loss_function(this_logits, this_y)
        if self.loss == "ce":  # loss: (n_sample, )
            loss = (loss * this_weight) / weight.sum()
        elif self.loss == "bce":  # loss: (n_sample, n_class)
            loss = (loss.sum(dim=1) * this_weight) / weight.sum()
        else:
            raise NotImplementedError
        loss = loss.sum() * self.factor
        total_loss += loss
    else:
        this_logits = logits[start:end]
        this_y = y[i]
        this_weight = weight[i]

        loss = self.loss_function(this_logits, this_y)
        if self.loss == "ce":  # loss: (n_sample, )
            loss = (loss * this_weight) / weight.sum()
        elif self.loss == "bce":  # loss: (n_sample, n_class)
            loss = (loss.sum(dim=1) * this_weight) / weight.sum()
        else:
            raise NotImplementedError
        loss = loss.sum() / self.factor
        total_loss += loss
Post Processing
Calculate a moving average with weights of [0.1, 0.2, 0.4, 0.2, 0.1] and finally add the global average * 0.2 for each species.

This consistently boosted public and private LB by 0.014~0.016 in both final submissions.

def smooth_array_general(array, w=[0.1, 0.2, 0.4, 0.2, 0.1]):
    smoothed_array = np.zeros_like(array)
    timesteps = array.shape[0]
    radius = len(w) // 2

    for t in range(timesteps):
        for i, weight in enumerate(w):
            index = t - radius + i
            if index < 0: 
                smoothed_array[t] += array[0] * weight
            elif index >= timesteps: 
                smoothed_array[t] += array[-1] * weight
            else:
                smoothed_array[t] += array[index] * weight
    for c in range(array.shape[1]):
        smoothed_array[:, c] = smoothed_array[:, c] * 0.8 + smoothed_array[:, c].mean(keepdims=True) * 0.2
    return smoothed_array
Other Setting
resampling became a bottleneck, so preprocess sampled to 32 kHz and storing them on disk helps speed up the training phase.
sampling by RMS (+0.010)
RMS sampling > using first 5sec > random sampling
backbone
resnet18d, resnet34d and efficientnetv2s
Thank you for reading

Project Details
Author
penguin46
ryotayoshinobu


Share
2
4 Comments
Hotness
Aaditya Porwal
Posted a year ago

25th in this Competition

Congrats!! for the 6th place.


Reply
RihanPiggy
Posted a year ago

7th in this Competition

Congratulations for the 6th place! I am happy that the training and inference tricks I used last year also help this year!


Reply
1
1
penguin46
Topic Author
Posted a year ago

6th in this Competition

@honglihang
Congratulations to you too on your gold medal! I was really helped by your last year's solution, so I am glad that we both stayed in the gold zone!


Reply
1
1
RihanPiggy
Posted a year ago

7th in this Competition

@ryotayoshinobu
Yes, this year's birdclef is really a hard one, glad to see that we both survived the shake with a relatively robust solution.
I tried moving average last year but it decreased LB in last year's metrics… emmmm, sad T T


Reply

5 more replies
Profile picture for RihanPiggy
Profile picture for Aphysict
Profile picture for penguin46
C R Suthikshn Kumar
Posted a year ago

27th in this Competition

Congratulations on achieving 6th place in this competition. Appreciation for sharing your solution details.


Reply
Skip to
content
Kaggle

Create
Search

Home

Competitions

Datasets

Models

Benchmarks

Game Arena

Code

Discussions

Learn

More


View Active Events

Skip to
content
Kaggle

Sign In

Register
7th Place Solution for the BirdCLEF 2024 Competition

BirdCLEF 2024

Solution Writeup · 7th place · Jun 11, 2024

Congratulations to all the winners! Thanks to Kaggle and Cornell Lab of Ornithology for hosting this interesting competition.

This year's birdclef is really a hard one. Unlike padded CMAP which suppress the impact of species with very few positive labels, AUC treat every species equally and is very sensitive.

I am happy to survive the final shake. Let me introduce my solution.

Thanks to every competitor who gave me inspiration. Special thanks to atsunorifujita, martynoveduard, loonypenguin, vladimirsydor, anonamename

Context
Business context: https://www.kaggle.com/competitions/birdclef-2024/overview
Data context: https://www.kaggle.com/competitions/birdclef-2024/data
Overview of the approach
My final submission is a combination of 2 SED model and 1 CNN model. The training scheme is almost the same as my last year's 2nd place solution.
The new stuffs I added include

Extract training sample from soundscape using Birdnet and Bird-vocalization-classifier
Knowledge distillation proposed by last year's 4th place
Sumixup proposed by last year's 7th place
Int8 quantize
Split species to serveral subset (most important for me)
The key to win this competition is to split species to several subset and train the subset separately, which enables model to focus more on each species compared to train them all.

My final submission is an ensemble of 2 models with all species and 1 model with 66 species which have relatively small sample size.
Actually I had an experiment which split species to 3 subset, which gave me 0.690440(1st place) in private LB, but it was too risky to choose because I was not sure whether I was just overfitting the public LB.

Details of the submission
Models
SED with tf_efficientnetv2_s_in21k (all species)
SED with seresnext26t_32x4d (66 rare species)
CNN with resnet34d (all species)
Extra data on xeno-canto
Like last year's situation, I collected extra audio from xeno-canto and trained a baseline SED v2s model. The baseline score was 0.65.

Extracting train sample from soundscape using Birdnet and Bird-vocalization-classifier
Birdnet covers 181 of 182 species and Bird-vocalization-classifier covers 180 of 182 species. I mainly used birdnet and extracted niwpig1 with bird-vocalization-classifier. I extracted the 15 second audio clip with threshold 0.3.

This boosted my baseline score to 0.68.

Knowledge Distillation
I implemented the same knowledge distillation scheme as last year's 4th place. This further boosted my baseline score to 0.70.

Further Extracting train sample from soundscape using trained models
I used my own trained models to further extract train samples from soundscape, this gave me a little boost in LB.

Species subset
Inspired by 3rd place in birdclef 2022, I tried to train a model with 66 species which have relatively small sample size, and I found that this model outperformed model trained with all species on 43 of 66 species.

I was not sure whether this will also happen in private LB, but I decided to ensemble 1 model trained on 66 species into my final submission.

I also tried to split species into 3 subsets with some overlap. I trained them separately and then created a submission with 3 subset model ensemble. This gave me 0.690440(1st place) in private LB, thus maybe splitting birds to subset is the key to win. But it was too risky to choose because I was not sure whether I was just overfitting the public LB.

Int8 quantize
Thanks to the unlabeled soundscape, we can perform quantization this year.

I used nncf to quantize encoder part of sed model. Quantizing CNN model led to 0.01 decrease in LB score, so I didn't perform quantization for CNN.

Quantization reduced inference time by 20 minutes for sed v2s model. (100 minutes to 80 minutes)

import nncf
import openvino as ov

nncf_dataset = nncf.Dataset(pytorch_dataset, transform_fn)
model = ov.Core().read_model(path_to_model)
quantized_model = nncf.quantize(
    model, nncf_dataset,
    target_device=nncf.TargetDevice.CPU,
    subset_size=300,
    fast_bias_correction=True,
    preset=nncf.QuantizationPreset.MIXED,
)

ov.save_model(quantized_model, quantized_model_path)
ensemble strategy
logit average of 3 models
rank average of 3 models
Things didn't work for me.
Too many…

Adding validation data to training led to 0.01 LB decrease. It is very weird, but I have to accept it and be faithful…
Calculating species weight to deal with label shift and covariance shift
Giving more weight to audios recorded in india according to geometric information (latitude 8~21, longitude 72.5 ~ 79)
changing weight of the sampler significantly decrease the LB. Sampling every species equally performed best
Filtering out 2024 train data duplicates not only by id but also by 'author + primary_label'(proposed here)
using embedding extracted from birdnet and bird-vocalization-classifier proposed by last year's 6th place
Sources
https://www.kaggle.com/competitions/birdclef-2023/discussion/412707
https://www.kaggle.com/competitions/birdclef-2023/discussion/412753
https://www.kaggle.com/competitions/birdclef-2023/discussion/412922
https://www.kaggle.com/competitions/birdclef-2023/discussion/412808
https://www.kaggle.com/competitions/birdclef-2023/discussion/412708
https://www.kaggle.com/competitions/birdclef-2022/discussion/327193
https://docs.openvino.ai/2023.3/basic_quantization_flow.html
Project Details
Author
RihanPiggy
honglihang


Share
12 Comments
Hotness
Aaditya Porwal
Posted a year ago

25th in this Competition

Congratulations for gold medal !!!!!!!!


Reply
RihanPiggy
Topic Author
Posted a year ago

7th in this Competition

Thank you!


Reply
Ayush Solanki
Posted a year ago

928th in this Competition

Congratulations for gold!! I Had a beginner level doubt like are you feeding the model single channel spectrogram like (1,256,625) example shape or 3 channel by stacking the single channel?


Reply
RihanPiggy
Topic Author
Posted a year ago

7th in this Competition

Thank you!
Mel spectrogram is single channel. And I used compute delta and stack to 3 channel.

def make_delta(
    input_tensor: torch.Tensor
):
    input_tensor = input_tensor.transpose(3,2)
    input_tensor = compute_deltas(input_tensor)
    input_tensor = input_tensor.transpose(3,2)
    return input_tensor


def image_delta(x):
    delta_1 = make_delta(x)
    delta_2 = make_delta(delta_1)
    x = torch.cat([x,delta_1,delta_2], dim=1)
    return x

Reply
1
Ayush Solanki
Posted a year ago

928th in this Competition

Thank you for response was searching for something like this from long time


Reply
Bartley
Posted a year ago

342nd in this Competition

Thanks for sharing your solution again, I learnt a lot from what you shared last year!

You mention “Sampling every species equally”. Does this mean that you randomly sampled a species with weight 1/182, then sampled a datapoint within that species, or does it mean that each data point was seen once each epoch?

Also, could you clarify how you create the 10-sec segment during training? Is the same as 2023 (eg. 60sec waveform -> split to 6 * 10sec -> np.sum(audios,axis=0))?


Reply
RihanPiggy
Topic Author
Posted a year ago

7th in this Competition

Thank you!.

Yes, the weight for each species is 1/182, but it is not a two step sampling. I used weighted sampler provided by pytorch. Let's assume there are 3 species (a,b,c). a has 3 samples, b has 2 samples, c has 1 sample. Then the weight for sample of a, b, c is 1/3, 1/3, 1/3, 1/2, 1/2, 1/1. You can find the weight for each species is the same here.

For SED model, I simply randomly selected 10 sec from the audio clip. For audio extracted from unlabeled soundscape( I clipped 15sec), it is 15sec waveform -> 10sec, 5sec -> 10sec, 10sec(pad 0) -> np.sum(audios,axis=0))


Reply
Bartley
Posted a year ago

342nd in this Competition

Got it. Thanks!


Reply
Donghui Zhang
Posted a year ago

11th in this Competition

Congratulations on winning the gold medal. I would like to ask why you thought of int8 quantization.


Reply
RihanPiggy
Topic Author
Posted a year ago

7th in this Competition

Thank you! The same to your team, Congratulations.
Well, my best single model's inference time is around 95min, which drove me to find some way to make it faster. And I came up with int8 quantization.
I searched the way to perform quantize and found that what you only need is a calibration dataset, and unlabeled soundscape provided by the host this year is just perfect for it!


Reply
Donghui Zhang
Posted a year ago

11th in this Competition

I understand, thanks for the reply!


Reply
This comment has been deleted.

Aphysict
Posted a year ago

9th in this Competition

Maybe reduce training epochs when training with full data. But yeah, no significant improvement from my experience.


Reply
RihanPiggy
Topic Author
Posted a year ago

7th in this Competition

I trained quite long epoch because I used knowledge distillation.
I think you are right, my CNN model shown significant LB decrease in long epoch…Sed model is more stable. Still, it is quite weird for me…


Reply
Aphysict
Posted a year ago

9th in this Competition

I only try cnn models. I gauss different models have different fitting capabilities. Even for cnn models, different backbones perform differently, some(rexnet) can have really high CV, yet low leaderboard scores.


Reply
RihanPiggy
Topic Author
Posted a year ago

7th in this Competition

The same. For me, tf_efficientnet_b3ns performed really bad this year. Maybe it fits species not contained in test set better.


Reply
Skip to
content
Kaggle

Create
Search

Home

Competitions

Datasets

Models

Benchmarks

Game Arena

Code

Discussions

Learn

More


View Active Events

Skip to
content
Kaggle

Sign In

Register
8th place solution

BirdCLEF 2024

Solution Writeup · 8th place · Jun 13, 2024

8th Place Solution
I would like to thank everyone who organized this great competition and all the participants who worked hard to complete it !!

Spectrogram
MelSpectrogram(
    sample_rate=32_000,
    n_fft=1_095,
    hop_length=500,
    f_min=40,
    f_max=15_000,
    n_mels=128,
)
Model
I used an ensemble of three models.

SED

eca_nfnet_l0
tf_efficientnet_b0.ns_jft_in1k
Simple 2D CNN

eca_nfnet_l0
logits ensemble seems slightly better than rank ensemble.

Inference
Used openvino to reduce inference time (thanks @honglihang to openvino notebook)
inference time: ~ 90 mins
Training
Loss
FocalLossBCE
It is almost the same as a great baseline(thanks @salmanahmedtamu)
Used secondary labels as 1.0
Step0. Pretrain
Data

2021, 2022, and 2023, but only those with primary labels that are not included in 2024's 182.
First 5 seconds
Details

Signal augmentations

NoiseInjection
GaussianNoise
PinkNoise
RandomVolume
Spectrogram augmentations

FrequencyMasking (2 bands)
TimeMasking (2 bands)
RandomFlip
Step1. Finetune for Pseudo-Labeling
As discussed extensively, there was a significant difference in distribution between the xeno data and the test data. This was evident from the model's preds, influenced by various factors like insect noise, bird volume, and echoes from obstacles. To address this gap, I used pseudo-labeling as one of the measures.

Data

Only 2024
Filtered by Google's bird vocalization classifier preds.
5 seconds
Details

Same augmentations as in pretraining
Simple mixup
I trained models for 15 epochs from a pretrained model and used the model with the highest CV score to infer on unlabeled data.

Step2. Finetune for Submission
Data

2024, ff1010bird (nocall only), unlabeled_soundscapes (with step 1 preds)
Filtered using Google's bird vocalization classifier pred scores to extract the foreground for all but nocall
5 seconds
full data
Details

sumix (ref: BirdCLEF2023 7th solution)
sumix is a simple yet highly effective.
Other Ideas Explored but Insufficiently Tested
Variants of SED (frequency-axis attention, hybrid, etc.)
1D models
Various mixup methods (sumix was the best)
longer duration (as a measure against weakly labeled data)
Background noise, heavy augmentations
Multi round pseudo labeling
TTA (time-shifted slightly)
KD
I experimented with various ideas, but most of them don't appear to be effective. However, I can't confidently assess their effectiveness because I couldn't establish a reliable CV scheme. Consequently, I had to rely on the leaderboard to assess their effectiveness. This has made BirdCLEF 2024 a particularly tough competition for me.

In the end, pseudo-labeling was my only method to bridge the domain gap, but after visualizing the pseudo-labels, it became clear that they were inaccurate. I still don't fully understand what was key, but I have a sense that it was beneficial to build a simple pipeline.

Inference notebook: https://www.kaggle.com/code/kapenon/bc2024-8th-place-solution-kapenon

Acknowledgement
I would like to express my sincere gratitude to Rist Inc. for their invaluable support.

Project Details
Author
kapenon
kapenon


Share
6 Comments
Hotness
C R Suthikshn Kumar
Posted a year ago

27th in this Competition

Congratulations on securing 8th place in this competition. Thanks for sharing the details of your models.
I observe you are using 32_000 instead of 32,000. Whether _ is ok to be in integer values.


Reply
kapenon
Topic Author
Posted a year ago

8th in this Competition

It follows Python syntax. The meaning is the same as 32,000.


Reply
Aaditya Porwal
Posted a year ago

25th in this Competition

Congratsss for gold!!


Reply
RihanPiggy
Posted a year ago

7th in this Competition

congratulations! Glad to see openvino still works well~


Reply
thacrobatheskis
Posted a year ago

308th in this Competition

Congrats! How did you use the google bird vocalization classifier to filter the training data & soundscapes? (I tried using the max class probability for it as a no-call classifier, but it didn't seem to work very well.)


Reply
kapenon
Topic Author
Posted a year ago

8th in this Competition

I referred to this Notebook (https://www.kaggle.com/code/philculliton/inferring-birds-with-kaggle-models) to perform inference, and used the max probability for each 5-second clip as the result of the call/no-call classifier.

Hmm, I should be doing almost the same process as you. Could it be the influence of combining other processes?　(e.g. adding no_call class)


Reply
thacrobatheskis
Posted a year ago

308th in this Competition

Interesting! Did you take the max over just the competition bird probabilities, or all birds? (I took it over all birds)
How did you use it on the train_audio? Some options: remove the audio from the train data, keep it but set the targets to 0 prob, or add a no_call class (I kept the data but set the targets to 0 prob if the google bird model predicted < threshold)
What threshold did you use?

Sorry for all the questions! I tried several variations, so super curious about your approach here :)


Reply
kapenon
Topic Author
Posted a year ago

8th in this Competition

I took the maximum over all birds (not 182), just like you. I assumed BirdCLEF2024 data only contained vocalizations from 182 bird species (though I'm not confident about the unlabeled ones), so I chose this approach to account for the possibility of misclassification between classes by Google's model, which has been trained on many classes.

The threshold is set at 0.5. In my training pipeline, I iterated over each audio file during training. Each audio file was split into 5-second clips, and I filtered all clips based on Google's model predictions using a threshold of 0.5. From those clips with predictions above 0.5, one was randomly selected. If no clips exceeded 0.5, I chose randomly from all clips.

I'm sorry for any misunderstanding caused by my previous responses. Let me clarify:

The "no_call" class I referred to is from the ff1010 dataset. My hypothesis was that filtering with Google's model would leave us with data where bird vocalizations are clearly present. To achieve both the addition of background noise and maintaining label accuracy, I used mixup to blend the data.


Reply
Skip to
content
Kaggle

Create
Search

Home

Competitions

Datasets

Models

Benchmarks

Game Arena

Code

Discussions

Learn

More


View Active Events

Skip to
content
Kaggle

Sign In

Register
9th place recap

BirdCLEF 2024

Solution Writeup · 9th place · Jun 11, 2024

Share some personal experiences in this competition.

CV strategy
No CV, only public leaderboard. It just works.
At early stage, I tried to build CV and even wasted lots of submissions to do leaderboard probing. Then I realized that CV is just waste of time. I just assume unlabeled soundscapes, public test set and private test set share similar data distribution later on.

Data Augmentation
Only cutmix and mixup. Only one model with time & freq mask and it overfits.
Traditional data augmentation tends to overfit the train data, because it teaches the model to output the same representation for data from single audio clip (single domain knowledge). While things like mixup have some cross domain knowledge, because data comes from different audio clips. (not really sure, just guessing)
nice pic from OPTIMAL REPRESENTATIONS FOR COVARIATE SHIFT

Some Model Design
Almost last year's model. Add a bottleneck layer to let the representation more compact.
self.bottleneck_layer = nn.Sequential( nn.Linear(backbone_out, cfg.bottleneck_dim), nn.BatchNorm1d(cfg.bottleneck_dim), nn.ReLU())
bottleneck_dim=1024(actually 512 is better). This can improve score a little(less than 0.01).

With eca_nfnet_l0 as backbone and above tips, my baseline public score is 0.65(close to 0.66).

Contrastive Adversarial Domain (CAD) bottleneck
from the paper OPTIMAL REPRESENTATIONS FOR COVARIATE SHIFT.
model
Honestly, I don't understand the math behind that(read the paper if you are interest). Practically, it use data domains as labels (in our case, train data: 0, unlabeled_soundscapes: 1), and bottleneck_layer's output as features to do SupCon learning. The total loss is traditional BCE loss + weight*SupCon loss. The weight and the temperature in SupCon loss is tuned base on public leaderboard.

weight	Public	Private
1e-2	0.56	0.57
1e-3	0.66	0.63
1e-4	0.68	0.66
1e-5	0.65	0.64
temperature	Public	Private
0.07	0.62	0.62
0.09	0.65	0.65
0.10	0.68	0.66
0.11	0.65	0.66
supcon loss

At the early stage, BCE loss dominates, then SupCon loss starts to kick in.

Post-process trick
Average prediction with previous and next window. Improvement depends(0.01~0.02).

Model ensemble and selection
My best ensemble is actually two eca_nfnet_l0s with different data split and SupCon loss weight, it scores 0.68 in private, but I didn't select it.
I further ensemble these two with a tf_efficientnetv2_b0(public 0.65, private 0.62, with time & freq mask) and rexnet_100(public 0.63, private 0.64). Don't have enough time to finetune these two, maybe they can improve.

Things didn't work(overfitting mostly)
pretrain (on 22&23 data, maybe I should try pretrain on unlabeled_soundscapes)
knowledge distillation (only try train data, maybe I should try knowledge distillation on unlabeled_soundscapes)
Entmin (definitely useless)
DANN (no improvement)
focal loss (it actually works solely, but no improvement with CAD)

Final thought
This competition is very hard. On domain adaptation/generalization leaderboard Domainbed, simple baseline is hard to beat. In practice, no current methods for learning representation uniformly outperform the baseline. I think that's reason for the big shakeup, and why public notebooks outperform carefully trained models.

Project Details
Author
Aphysict
aphysict


Share
4
6 Comments
Hotness
Tom Denton
Competition Host
Posted a year ago

Cool approach! Really nice to see someone try a domain adaptation method in the competition context.


Reply
Aphysict
Topic Author
Posted a year ago

9th in this Competition

Thanks! Glad you like it.


Reply
Eiichi Okumura
Posted a year ago

403rd in this Competition

Congratulations on your achievement! Thank you for your detailed explanation with diagrams.

This was my first time participating in a Kaggle competition, and through the experience, I developed a strong interest in domain adaptation. You mentioned that you successfully used CAD to address domain shift, while DANN was not as effective. How do you approach the selection of methods for domain adaptation?

I have tried three domain adaptation methods: Cycle GAN, DANN, and M2D. The top private scores in my submissions were achieved using Cycle GAN, suggesting some level of effectiveness, but DANN did not perform well.

Despite the significant shake-up at the end, I am determined to learn and improve. Thank you for your time and insights.


Reply
Aphysict
Topic Author
Posted a year ago

9th in this Competition

I can recommend a reading material which leads me to the CAD paper. As for method selection, just trial and error. Like I said, no current methods uniformly outperform the baseline. The downside of DANN is, ''(i) it
maximizes an upper bound on the desired term;
(ii) it requires adversarial training, which is challenging in practice'', quoted from CAD paper. You should also try DomainBed which contain lots of domain adaptation tricks.


Reply
Eiichi Okumura
Posted a year ago

403rd in this Competition

Thank you for your response and the recommendation of reading material that led you to the CAD paper. I will definitely look into it.

As for the method selection, I appreciate your insight about trial and error. It's reassuring to know that even experienced practitioners face challenges with adversarial training. I found the quote from the CAD paper about the downsides of DANN particularly enlightening.

I will also explore DomainBed as you suggested. It's good to know that it contains various domain adaptation tricks that could be useful for my future projects.

Thank you once again for your guidance and recommendations. Your insights are incredibly valuable to someone new to these methods.


Reply
Aaditya Porwal
Posted a year ago

25th in this Competition

Congratsss!!!


Reply
C R Suthikshn Kumar
Posted a year ago

27th in this Competition

Congratulations on achieving 9th place in this competition. Thanks for sharing your insights.


Reply
Skip to
content
Kaggle

Create
Search

Home

Competitions

Datasets

Models

Benchmarks

Game Arena

Code

Discussions

Learn

More


View Active Events

Skip to
content
Kaggle

Sign In

Register
10th Solution

BirdCLEF 2024

Solution Writeup · 10th place

Congrats to all the winners, and thank you to Kaggle and Cornell Lab of Ornithology for organizing such an interesting competition. I will mainly explain the method from my solution that I believe helped improve my LB position.

Summary
I trained the model in three stages. First, I used all the data from 2021 to 2024. Next, I trained on either just the 2024 data or on the data from 2021 to 2024, focusing on the target of this competition. Finally, I trained on the second set of data, adding the unlabeled data from 2024. The final stage of training improved the model's performance on both the public and private LB.

Data
2021,2022,2023,2024

Model
tf_efficientnetv2_b0, b3 (timm)
input: single-channel spectrogram

Use of Unlabeled Data
I created pseudo-labels for the unlabeled data and used them by adding them to the labeled data with a 50% probability. This method was inspired by the 7th solution in 2023 and implemented with reference to GitHub. tgt_wav was the unlabeled data, add_wav was the labeled data, and db was a random positive value(~50). The labels were simply combined.

def __add_noise(self, tgt_wav, add_wav, db):
    tgt_rms = np.sqrt(np.mean(np.square(tgt_wav), axis=-1))
    add_rms = np.sqrt(np.mean(np.square(add_wav), axis=-1))

    noise_rms = tgt_rms / (10**(float(db) / 20)) 
    new_wav = tgt_wav + add_wav * (noise_rms / (add_rms + 1e-6))
    new_wav = np.clip(new_wav, np.min(new_wav)*2, np.max(new_wav)*2)
    return new_wav
I divided each 4-minute unlabeled data file into 5-second segments, creating 48 files from one unlabeled file.　After generating pseudo-labels for each segmented file, I used the top 10% of those with the lowest class label entropy for training. Labels below the top 90% for each class's pseudo-label were set to 0. The labels were used as soft labels. This method improved the LB by ~0.04.

Project Details
Author
Tamo
yuyuki11235


Share
1
4 Comments
Hotness
German Arley
Posted a year ago

Congratulations on securing 10th Place in this competition. Thanks for sharing details.


Reply
Mubashir Ahmed Siddiqui
Posted a year ago

Congratulations on securing 10th place in this competition, @yuyuki11235


Reply
C R Suthikshn Kumar
Posted a year ago

27th in this Competition

Congratulations on securing 10th Place in this competition. Thanks for sharing details.


Reply
clora(tnaka)
Posted a year ago

Congratulations solo gold medal.


Reply
Skip to
content
Kaggle

Create
Search

Home

Competitions

Datasets

Models

Benchmarks

Game Arena

Code

Discussions

Learn

More


View Active Events

Skip to
content
Kaggle

Sign In

Register
1st Place Solution: Multi-Iterative Noisy Student Is All You Need

BirdCLEF+ 2025

Solution Writeup · 1st place · Jun 15, 2025

I know that Kagglers are aware of what’s happening in Ukraine, and you’re here to take a look at my solution, but let me share a small glimpse from the competition’s deadline day that reflects our current reality: I had to make my final submissions from a shelter while we had a stable connection, as Ukraine was under another devastating attack with many civilian casualties, some of them from my neighborhood.
I would like to thank the Armed Forces of Ukraine, Security Service of Ukraine, Defence Intelligence of Ukraine, and the State Emergency Service of Ukraine for providing safety and security to participate in this great competition, complete this work, and help science, technology, and business not to stop but to move forward.

TLDR
SED models on 20-second input chunks.
A Multi-Iterative Noisy Student is used as a self-training approach via MixUp between focal training data and pseudo-labeled soundscapes.
Power transform applied to pseudo-labels to reduce noise.
Pseudo-label sampler assigns weights equal to the sum of the maximum of labels within each soundscape.
A separate model for Amphibia and Insecta label groups using extended species data from Xeno-Canto.
A final ensemble with models from different training iterations.
Inference is performed by averaging overlapping framewise predictions from neighboring chunks, followed by smoothing and delta shift inference.
Solution Overview


Data
Additional Xeno-Canto Data
Target species
Num samples: 5489
Samples per species groups: Aves(birds)=5480, Amphibia=6, Mammalia=3
Max samples per species: 500
Comment: It usually worsened results, so only one model saw that data.
Extra species
Num samples: 17197
Samples per species groups: Insecta=16218(544 extra species), Amphibia=979(113 extra species)
Max samples per species: 200
Additional filters: duration less than 60 sec
Comment: This data was used to train a separate dedicated model for the Insecta and Amphibia groups.
Interesting note about Insecta
The Insecta group included labels at the family level, such as Cicadidae, Gryllidae, and Tettigoniidae, while the rest of the Insecta labels were species from the Tettigoniidae family. But the following experiments showed that the family-level labels were probably related to the specific species or at least to a narrow set of species that inhabit the Middle Magdalena Valley:

Including the Tettigoniidae label in secondary labels of species from the Tettigoniidae family worsened results.
Including extra Xeno-Canto data for Gryllidae and Tettigoniidae (i.e., family-level labels) in training on target species also worsened results.
Using additional Gryllidae and Tettigoniidae samples from Xeno-Canto, assigning them unique new labels based on the species of each sample instead of assigning the family-level labels, and training a dedicated model improved results. It means that Tettigoniidae and Gryllidae target labels are related to the specific species that can be separated from other species within these families that are present as other Insecta target labels or extra species from these families that were downloaded.
Data preparation
5 folds
Each fold includes at least 1 sample for each label
20-second audio chunks normalized by absmax
All secondary labels = 1
From the start, I came to the thought that the presence of Amphibia and Insecta groups would make models favor longer input durations over shorter ones, since the long duration and repetitiveness of their calls are distinctive features for species from these groups.
To find the most optimal duration, I conducted multiple experiments with different durations and figured that 20-second chunks work best for me, and any longer duration did not improve results but took more time to infer, so I continued with that chunk duration.
Here are the Public scores that I obtained by experimenting with different durations while training(supervised with train data only) an ensemble of 5 SED efficientnetb0 models and adjusting proportionally spectrogram hop length(more detailed information on the models and inference is provided in the next sections):

Chunk duration	Public LB
5 sec	0.842
10 sec	0.864
15 sec	0.87
20 sec	0.872
30 sec	0.872
Models
Architectures
Across different training stages, I used various CNNs, gradually incorporating more complex models at each stage. All models included the SED head (adaptation from the 4th place 2021 ), which consistently provided a significant boost over other head variations.

SED
Gem frequency pooling
Repeated 3 Mel Spectrograms as input
1 stage backbones:
tf_efficientnet_b0.ns_jft_in1k
regnety_008.pycls_in1k
1 pseudo-labeling iteration backbones:
tf_efficientnet_b0.ns_jft_in1k
regnety_008.pycls_in1k
tf_efficientnet_b3.ns_jft_in1k
regnety_016.tv2_in1k
2-4 pseudo-labeling iteration backbones:
tf_efficientnet_b3.ns_jft_in1k
tf_efficientnet_b4.ns_jft_in1k
regnety_016.tv2_in1k
eca_nfnet_l0.ra2_in1k
Amphibia/Insecta model backbone:
tf_efficientnet_b0.ns_jft_in1k
Mel spectrogram parameters
20 sec -> Image size = (3, 224, 512)
MelSpectrogram (sample_rate: 32000, mel_bins: 224, fmin: 0, fmax: 16000, n_fft: 4096, hop_size: 1252, top_db=80.0)
0-1 normalization
Thoughts on mel parameters tuning
Because long input chunks were used, I had to set a larger hop length value, otherwise, inference would take too long, and I wouldn’t have the capacity to prepare a good ensemble.
The important thing was setting a larger number of n_mels. I suspect this is because some species (especially from the Amphibia and Insecta groups) have calls within narrow frequency ranges, so showing more mel bands to models was important to distinguish species well.
Validation
I did not find a good CV/ LB correlation, so considering the Host's words that public/private distributions are very similar and the knowledge from last year's solutions, I validated ideas using only the public LB.
A single model’s LB score varied a lot for different seeds, so usually I trained the same setup on different folds(2-5, depending on the remaining time that I had) and ensembled them to obtain a more reliable response from the LB.
As it turned out, the Hosts were absolutely honest with us(did not have any doubt), and most Public results correlated pretty well with the private LB.

1 Stage (Supervised Learning)
Training Details
Epochs: 15
Loss: CrossEntropy
LR: 5e-4 - 1e-6(same for all models)
Optimizer: AdamW with 1e-4 weight decay
Scheduler: CosineAnnealingWarmRestarts with restart after each 5 epochs. Applying warm restarts, I could train longer than with one cycle
BS: 64
Augmentations:
Mixup: p = 0.5, on normalized by absmax raw audio with an equal sampling weight for each species
Padding: To keep samples of different lengths overlapping after mixup, the left part of it was filled with 0, so on the right, there is always an overlap that ensures train samples are always actually mixed up
Models: efficientnet0, regnety8. An ensemble of small models on the 1st stage gave almost the same results as ensembling deeper ones
Loss choice
I noticed that the choice of loss got a lot of attention in the discussions. So I conducted some experiments and found that both CE and BCE/Focal losses could give me similar results when the learning rate and the number of epochs were well-tuned. However, CE gave me a bit better results, so I settled on it.
I connect better results with CE (I might be wrong) with the following interconnected assumptions:

The magnitude of updates with CE for each label depends on how well the positive labels (probability > 0) are classified. This means that if a rare positive label A gets a low probability, then the negative overrepresented label B(that has a higher probability than other negative labels) is pushed to zero with a stronger update.
CE handles imbalanced labels better and avoids overfitting to overrepresented classes by punishing them when Softmax can not give a higher score for A because the overrepresented label B already got too high logits as a result of the previous numerous imbalanced updates when label B was positive.
Also, I didn’t normalize sample labels to sum to one, motivated by the idea that more difficult samples (those with more positive labels) should have a greater impact on the loss.

Inference


I am showing my inference flow first to give more context before going to the next sections, where I explain how I used pseudo-labels, which were generated with that inference. I hope my diagram does not look too overloaded.
The core idea was to fully leverage all framewise predictions produced by the SED head by averaging the overlapping framewise predictions from neighboring audio chunks, rather than taking max only from the central 5 sec and throwing away precious predictions.

It can be seen as a 1D analogue of 2D sliding-window segmentation of large images, rather than treating each audio chunk as a completely separate sample.
It can be seen as a form of test-time augmentation (TTA) because each framewise prediction is averaged over multiple chunks that present the same time frame to the model with slightly different surrounding context.
Consistently boosted my LB score (by 0.002-0.003).
Helped to get more generalizable predictions.
Other inference/postprocessing tricks
Padded the left and right sides of the signal to ensure that the first and last 5-second chunks are centered after splitting. Framewise predictions related to the padding were then removed.
Smoothing [0.1, 0.2, 0.4, 0.2, 0.1].
Delta shift TTA (from the 2nd solution 2023).
Self-training
After hitting the ceiling with the supervised approach, it became clear that further improvements would come with the usage of the unlabeled soundscapes.
I pseudo-labeled the unlabeled data using the best LB ensemble from the 1 stage with the described above inference flow, and began experimenting with the ways to incorporate pseudo-labeled data in the training.
Initial attempts to concatenate the pseudo-labeled data into training batches separately didn’t succeed.

MixUps (finally working)
Then I tried mixing up the pseudo-labeled raw data with the training raw data, following the approach from the 2nd place solution 2024. At first, it didn’t work because I had set the Beta distribution’s parameters to be too low(used to sample blending weights). But after switching to a constant blending weight of 0.5(Beta’s parameters = inf), the magic started to happen, and the LB score began to rise. I suspect that is related to the fact that blending weights far from 0.5 sometimes suppress the meaningful signals, especially when mixing relatively clear train data with much noisier train soundscapes.

Stochastic Depth
Reading the paper on the Noisy Student approach [1], I found certain similarities with the self-training approach I was using.
So, I started experimenting with the techniques that showed a positive impact on self-training in the mentioned paper and found that adding Stochastic Depth [3] (i.e., dropout applied to the entire residual blocks) worked for me as well.

Applying drop_path_rate = 0.15 (which turned out to be optimal for all the models I trained), I consistently saw a boost in the LB score, up to 0.005 for some models.
Applying Stochastic Depth during supervised training didn’t lead to any improvement, which supports the idea that the current self-training approach is a form of Noisy Student self-training.
Why Noisy Student? (my understanding)
After reading [1], I finally understood why mixup works while simple concatenation doesn’t. Since it closely matched my approach, I considered it a form of Noisy Student self-training and named my solution accordingly.
Putting simply - showing to the model the same input and asking for the same output does not teach the student anything new, so in the best case it converges to the same results, in the worst case it accumulates error and the LB score drops (what I experienced). But when we inject noise(augmentations like mixup, drop paths) and ask to provide the same output as for the clean input, it starts learning more robust features instead of accumulating error.
I imagined the following scenario: we have a pseudo-labeled sample where A is the true label, and B is the negative one. The teacher model predicts A ≈ 1 and B ≈ 0, but not exactly zero. If we repeatedly train on this same input, the model may start learning irrelevant features associated with B, simply because its score isn’t exactly zero. It may also overfit to some noise because of thinking that it is related to A. At the same time, it doesn’t offer anything new beyond what the teacher model has already seen and learned, so it doesn’t lead to any improvement.
However, if we augment that sample, the student model must work harder to understand why the teacher gave a high score to A(an unaugmented input for the teacher). As a result, through this noisy student training, we force the model to focus on the most consistent, generalizable features relevant to A rather than memorizing noise associated with A and B.
Also, Noisy Student methodology involves including labeled samples in training (as I did), whose signals help guide the model toward better optima, especially during the early epochs.
And MixUps with pseudo-labeled data serve as a great augmentation for labeled samples, providing target domain backgrounds with soft-labels for possible species in that background.

Pseudo-labels preparation
Pseudo-labels were generated using the best ensemble of models from the previous stage.
Pseudo-labels were generated before self-training and stored as max label probabilities for 5-second segments, or framewise predictions were stored as they are without pooling(4 frames per 5-second segment).
Framewise predictions provided more splits of soundscapes into chunks(9 splits for 20-second chunks when save for each 5 sec, and 45 splits when save framewise predictions), but more splits usually did not provide better results
Pseudo-labeled data sampling
Soundscapes with a higher sum of maximum label probabilities were usually pseudo-labeled more accurately. This is because most soundscapes were overloaded with various species calls, and a low sum often indicated that the models struggled to recognize and distinguish those species.
WeightedRandomSampler was used with weights equal to the sum of maximum label probabilities within each soundscape. This ensured that samples with more accurate pseudo-labels were sampled more frequently. Idea to use a sampler for pseudo-labels, I found in that paper [2].
A random 20-second interval was selected from the training soundscape that was sampled by WeightedRandomSampler.
For that interval, the maximum probability for each label was taken across the 4 segments (or 16 frames), and then that soft labels were used in self-training.
WeightedRandomSampler stabilized training and boosted the LB score.
This approach was especially relevant after I reduced label noise (described in Multi-Iterative pseudo-labeling section) and ended up with many samples that had low label sums (less than 0.5 summing 206 labels), which was the same as using unlabeled data, so it was beneficial to give lower weights in sampling for such "almost" unlabeled samples.
Training Details:
More epochs = 25-35.
Drop path rate = 0.15.
Random padding. Samples shorter than 20 sec were placed at random positions within 20 sec.
Other training parameters are the same as in supervised learning.
Ratio of pseudo-labeled mixups
The ratio of labeled train samples that I mixed up with pseudo-labeled chunks in each batch(bs=64) was very important and significantly impacted the LB score.
To find the optimal ratio, I was gradually increasing it by 0.25, retraining an ensemble of 5 SED efficientnetb0 folds with the self-training setup described above, and checking the LB to find the best ratio.
Ratio of mixed samples	Public LB
0 (labeled training data only)	0.872
0.25	0.883
0.5	0.887
0.75	0.89
1.0	0.898
Turned out that mixing every training sample with a random pseudo-labeled sample showed the best score.
Multi-Iterative pseudo-labeling
Inspired by the 2nd place solution 2024 and papers on self-training [1, 2], I tried to pseudo-label the unlabeled soundscapes again, using models that were already trained on the pseudo-labels from the previous iteration. However, this approach didn’t work out of the box, and I spent some time investigating why until I found the correct preprocessing of the pseudo-labels that allowed me to keep training models on the next pseudo-labeling iterations.

Multi-Iterative labels preprocessing
The reason the models failed to converge in later iterations was that the pseudo-labels had become too noisy, obscuring any meaningful signal.
Here is the method that worked best for me and allowed me to move forward:

The trick was to apply a power greater than 1 to the probabilities (similar to temperature scaling, but applied to probabilities instead of logits). This worked because applying temperature to logits increases probabilities above 0.5, which was deteriorating.
Applying the power to the probabilities, I was able to preserve the important signals while preventing the amplification of confident noise.
In the image below, you can see that the power transform is a real power in the fight against label noise. After the transformation, labels become much cleaner, and only initially confident labels survive, having still pronounced values to train models.


Knowing how to train a multi-iterative noisy student, I ran 4 iterations, adjusting the pseudo-label power at each stage by validating results on the LB and consistently got the LB boost.
That self-training magic stopped working on the 5th pseudo-labeling iteration. I could not achieve any further improvement, so I stopped with the attempts to make new iterations work.

Iteration	Power value	Public LB
1	1	0.909
2	1 / 0.65	0.918
3	1 / 0.55	0.927
4	1 / 0.6	0.93
Training Details
Besides extending the training ensemble with eca_nfnet_l0, efficientnet4, and applying power to labels, all training details remained unchanged since the 1st iteration.
Separate model for Amphibia and Insecta
Knowing that species groups like Amphibia and Insecta are very underrepresented, and that Xeno-Canto provides data for multiple species from those groups that are not present in the train, I decided to try training a separate model that would see samples only from those groups, with a much higher diversity of species. The motivation was that the model would learn more representative features relevant to those groups, which would be a great supplement to the models that are trained only on a restricted number of samples/species from the training data.

Data details
Species groups: amphibia, insecta
Total number of species: 700
Total number of samples: 17844
Data sources: train, xeno-canto samples that are shorter than 1 minute
Minimum number of samples per species: 1 (raising that value to 5 dropped scores a lot)
Training Details
Epochs: 40
BS: 128 (with lower value scores dropped)
Model: efficient net 0 ns (deeper models and ensembles did not work much)
Other parameters are the same as for other models from my solution
Inference Details
Run inference for all species
Insert predictions for target species into a zero matrix, only their columns are non-zero, for ensembling with other models
After finding the optimal working parameters, I achieved a 0.002–0.003 LB boost.

Final Ensemble
I found that ensembling models from different training stages was beneficial not only for slightly improving the LB score, but also for giving me a little confidence that my solution is not overfitted by doing more and more self-training iterations.

The final ensemble consisted of the following 7 models trained on the specific data and different self-training iterations:

1 efficientnetb4 from 3rd self-training iteration
1 efficientnetb3 from 3rd self-training iteration
2 regnety016 from 4th self-training iteration
1 ecanfnetl0 from 3rd self-training iteration(with additional Xeno-Canto data for target species)
1 regnety008 from 1 stage(supervised training)
1 efficientnetb0 (supervised training on the extended Amphibia/Insecta species)
In my best solution, I tweaked the ensembling weights a bit, giving to efficientnetb3 and ecanfnetl0 slightly higher weights since they performed better as single models. However, the best private submission, with a score of 0.935, was achieved when I assigned equal weights to all models.

I believe that the ensembling of multi-stage models, diverse backbone architectures, and the dedicated model to certain species groups were crucial parts of my solution to withstanding the shake-up.
As a result, my best public LB score dropped only slightly on the private LB from 0.933 to 0.930.

Inference optimization
OpenVINO inference engine without quantization
Multiprocess loading of the test soundscapes
Spectrograms were generated once and then reused across all models
Closing words
I apologize that my write-up turned out to be a bit long. I did not want to skip anything important.
My Google Sheet accumulated during the competition over 320+ rows of ideas to check (95% of which were eventually marked red). So I decided not to include the What did not work block, otherwise it would significantly increase the length of the already long write-up.
Please feel free to ask me any questions about my solution in the comments. I will do my best to answer them.

I also want to thank all the participants of the previous BirdCLEF iterations who shared their ideas.
Without your brilliant ideas, I would not have been able to achieve that result.

Special thanks to the organizers, hosts, and everyone involved in conducting BirdCLEF. It is very appreciated that you put so much effort into conducting BirdCLEF each year, that you constantly stay in touch, and make each new iteration special(which is reflected in this year's number of participants).

References
[1] Self-training with Noisy Student improves ImageNet classification
[2] Design Choices for Enhancing Noisy Student Self-Training
[3] Deep Networks with Stochastic Depth

Resources
Inference notebook: https://www.kaggle.com/code/nikitababich/birdclef2025-1st-place-inference
Datasets: Extra Xeno-Canto data used in the solution

Project Details
Author
Nikita Babych
nikitababich


Share
19
18
14
35 Comments
6 appreciation comments
Hotness
Sarah Arshad
Posted 6 months ago

Great… what a remarkable work!


Reply
Devendra Bhaginath Jadhav (B21AI013)
Posted 6 months ago

Amazing Bro!


Reply
aaaditya56
Posted 6 months ago

Congrats! It's great to see a winning solution, as someone who had recently started with Kaggle. It provides more motivation to grow, and learn from this community.


Reply
Ten-Mergen Gandavaa
Posted 6 months ago

Congrats!!That was really helpful


Reply
zhuzhu521
Posted 6 months ago

Awesome! what a remarkable work！


Reply
Aniket Potabatti
Posted 6 months ago

Congratulations on an incredible 1st place solution, Nikita! This is truly impressive work and a fantastic write-up. Thank you for sharing such a detailed approach.


Reply
Khushi Yadav
Posted 6 months ago

Congratulations👏 for the amazing feat winning this competition!✨


Reply
Yassine Alouini
Posted 6 months ago

372nd in this Competition

Well done and thanks for sharing those insights! The iterative steps are very interesting.


Reply
Lyndon Reid
Posted 6 months ago

Congratulations and thank you for sharing Nikita, did you experiment with fmin and fmax at all?


Reply
Nikita Babych
Topic Author
Posted 6 months ago

1st in this Competition

Thank you @lyndonreidd
It is a good question.
Initially, I began with more "usual" mel parameters like MelSpectrogram(sample_rate: 32000, mel_bins: 128, fmin: 40, fmax: 15000, n_fft: 2048, hop_size: 512, top_db=80.0) and then step-by-step I tuned them based on the responses from the Public LB.
Talking specifically about fmin, fmax - adjusting them from (40, 15000) to (0, 16000) showed a very minor impact on the LB, so I did not play much with them after that.
I experienced a much more significant impact on the score by adjusting the following parameters: mel_bins, n_fft, hop_size ( you can check my explanation on why these adjustments were important in Thoughts on mel parameters tuning of my write-up).


Reply
Matheus Parracho
Posted 6 months ago

1171st in this Competition

amazing! congratulations!


Reply
Kurise
Posted 5 months ago

525th in this Competition

Hoping good luck with you and your country. I looked your solution on EEDI, hoping world to be peace.


Reply
Glin
Posted 5 months ago

504th in this Competition

@nikitababich Thank you again for your fantastic BirdCLEF 2025 solution write-up! I'm trying to replicate parts of it and have a few very specific questions about your Stage 1 (Supervised Learning):

Question 1: Gradient Clipping (max_grad_norm)

During Stage 1 training, did you use gradient clipping (i.e., set a max_grad_norm value)?
If yes, what value did you use for max_grad_norm?
Question 2: Loss Function Structure (Stage 1)

You mentioned using CrossEntropy (CE) loss for Stage 1. My current understanding for implementing a combined loss for an SED model is as follows. Could you please let me know if this aligns with your approach for Stage 1, or how your CE loss structure differed?

def compute_loss(self,outputs,labels):
    losses = {}
    losses["loss_clip"] = self.criterions["classification_clip"](
        torch.logit(outputs["clipwise_prob"]),labels
    )
    losses["loss_frame"] = self.criterions["classification_frame"](
        outputs["segmentwise_logit"].max(2)[0], labels
    )
    losses["loss"] = 0.5*losses["loss_clip"] + losses["loss_frame"] * 0.5
    return losses
Question 3: Stage 1 MixUp Padding Code

For the MixUp augmentation in Stage 1, you mentioned: "Padding: To keep samples of different lengths overlapping after mixup, the left part of it was filled with 0, so on the right, there is always an overlap that ensures train samples are always actually mixed up."
Could you please share the code that specifically implements this left-padding strategy for samples of varying lengths before the MixUp operation during Stage 1 training?
Your clarification on these specific points for Stage 1 would be extremely helpful.
Thank you for your time and willingness to share!


Reply
Nikita Babych
Topic Author
Posted 5 months ago

1st in this Competition

Hi @lhr124578
1) Regarding Gradient Clipping - I was checking gradient norms during training and did not see any dangerous fluctuations, so I stepped back from using it.
2) Your loss implementation is the one that I used.
3) Here is my padding function:

def pad_wave(wave, expected_len, pad_type="random"):
        wave_init_len = wave.shape[0]

        if wave_init_len >= expected_len:
            return wave[:expected_len]

        pad_len = expected_len - wave_init_len

        if pad_type == "random":
            padded_wave = np.zeros(expected_len, dtype=wave.dtype)
            insert_wave_start = np.random.randint(0, pad_len + 1)
            padded_wave[insert_wave_start: insert_wave_start + wave_init_len] = wave
            return padded_wave

        if pad_type == "left":
            return np.pad(wave, ((pad_len, 0)))

        elif pad_type == "repeat":
            reps = int(np.ceil(expected_len / wave_init_len))
            repeated = np.tile(wave, reps)
            return repeated[:expected_len]

        else:
            raise ValueError(f"Unsupported pad_type: {pad_type}")

Reply
Glin
Posted 5 months ago

504th in this Competition

@nikitababich
Thank you so much for taking the time to reply and for your incredibly helpful clarifications on my previous questions regarding gradient clipping, the loss function structure, and the MixUp padding in your Stage 1 setup! Your detailed explanations are invaluable.

I have a couple of quick follow-up questions based on re-reading your solution and general data processing considerations:

Human Voice Removal: Regarding the audio data (both the focal training data and the soundscapes used for pseudo-labeling, which you mentioned were normalized by absmax before MixUp), I noticed there wasn't a specific mention of further pre-processing steps like human voice removal .Could you perhaps share your thoughts or experience on this?

MixUp Target Handling: When applying MixUp to the absmax normalized audio waves (as mentioned for Stage 1), how were the corresponding target labels handled?

Were the target labels also mixed using the same lambda sampling weight? For example:
mixed_target = lambda * target1 + (1 - lambda) * target2
Or was there a different strategy for combining the targets after mixing the audio?

Reply
Nikita Babych
Topic Author
Posted 5 months ago

1st in this Competition

@lhr124578
1) I did not succeed with the preprocessing of CSA, and any attempt to remove human voice deteriorated the results. I hypothesize that most of the underrepresented species, whose samples are extremely noisy, are not present in the hidden data. So, when we filter out noise (human voice) from the samples of those species (which make up most of the duration of those samples), we are left with very short single calls that lack much of the surrounding context. These short calls are indistinguishable from parts of calls of other, more represented CSA species, making the preprocessing detrimental to training on those other species.

2) The maximum between targets was taken in all MixUps, whether it was train + train or train + pseudo.


Reply
Victor Caquilpan
Posted 5 months ago

719th in this Competition

Thanks @nikitababich, a full engineering work!


Reply
Harsh Gupta
Posted 6 months ago

This thing is really helpful and thanks for sharing this.


Reply
1
Bilal A. Chaudhry
Posted 6 months ago

1622nd in this Competition

Congratulations on this well deserved win, the quality of work on display is remarkable. Nailing a win with this wide a margin especially while under extreme duress took a level of courage and fortitude that is rarely seen.


Reply
SeshuRaju 🧘‍♂️
Posted 6 months ago

175th in this Competition

@nikitababich Congrats and you lead the Leaderboard in top almost 95% of the 3 months. New learnings from your solution - will try with late submissions.

Is it possible to share your experiments, which help as learning lesson to approach this competition of your My Google Sheet accumulated during the competition over 320+ rows of ideas to check (95% of which were eventually marked red).

Waiting for you Github, Congrats once again with stable CV.


Reply
Nikita Babych
Topic Author
Posted 6 months ago

1st in this Competition

Thank you @seshurajup
That table is very messy, I do not think it can be understood as it is now:) But if you think that the chronology and progression of those ideas might be useful, I'll consider polishing and sharing that sheet.


Reply
SeshuRaju 🧘‍♂️
Posted 3 months ago

175th in this Competition

Hi @nikitababich, any plans for release the github with your experiments?


Reply
Corentin Lobet
Posted 6 months ago

1672nd in this Competition

Awesome! what a remarkable work. Be safe 💙


Reply
1
Aayush Kumar Singha
Posted 6 months ago

313th in this Competition

Congratulations for the amazing feat winning this competition!!
Not to forget, winning in such adverse situation makes the win even more grand.
My constant prayers for yours' and your families' safety.


Reply
2
Tom Madden
Posted 5 months ago

Thanks so much for sharing. My first time digging into competitive architectures in one of these competitions and I learned a ton!


Reply
Shashi1024
Posted 5 months ago

1768th in this Competition

Congratulations and thanks for sharing your work


Reply
Dasun Weerakoon
Posted 5 months ago

Amazing Work!!


Reply
Hamza Bashir
Posted 5 months ago

Amazing Work!!


Reply
Pablo Yelloworld
Posted 5 months ago

1269th in this Competition

Huge congrats on your first-place win! What you pulled off is even more amazing given the incredibly tough situation you were in.

Thanks for sharing how you did it. Your insights are super valuable to the community and will definitely help everyone learn and grow.

I was really moved by what you shared about your reality during the competition. Finishing your final submissions from a shelter during an attack? That shows incredible grit and determination. It's a powerful reminder of how resilient people can be, pushing for excellence even when things are incredibly hard.

Your shout-out to the Armed Forces of Ukraine, Security Service, Defence Intelligence, and State Emergency Service really highlights the crucial support that made your participation possible. Their work keeping people safe and secure didn't just protect lives; it also let you contribute to science and tech progress despite the ongoing conflict.

Your story is a powerful testament to simply not giving up. Thanks for not only sharing your technical know-how but also for giving us this important look at your journey to victory.


Reply
102203848_Krish_Kumar
Posted 5 months ago

Amazing Work!!


Reply
This comment has been deleted.

Skip to
content
Kaggle

Create
Search

Home

Competitions

Datasets

Models

Benchmarks

Game Arena

Code

Discussions

Learn

More


View Active Events

Skip to
content
Kaggle

Sign In

Register
2nd Place. Journey Down the Rabbit Hole of Pseudo Labels

BirdCLEF+ 2025

Solution Writeup · 2nd place · Jun 12, 2025

We would like to thank the Armed Forces of Ukraine, the Security Service of Ukraine, the Defence Intelligence of Ukraine, and the State Emergency Service of Ukraine for providing safety and security to participate in this great competition, complete this work, and help science, technology, and business not to stop but to move forward.

As the competition was coming to an end, russia once again launched a massive missile and drone attack on peaceful Ukrainian cities. This terrorist act targeted civilian homes and critical infrastructure, resulting in widespread destruction and civilian casualties. Moreover, the use of the “double tap” tactic—striking the same location twice to kill rescuers aiding those trapped in damaged and destroyed buildings—further highlights the inhumanity of the assault. This action once again affirms russia’s status as a terrorist state.

Opening Words
If we decompose the entire BirdCLEF competitions journey, we can say that up to and including 2023 were the times of “Additional Data is All You Need” but since 2024, the paradigm has changed into “Journey Down the Rabbit Hole of Pseudo Labels”. So let’s jump into it and explore the Wonderland of Semi-Supervised Learning.

Short Data Story
We downloaded additional Xeno-Canto data, taking into account the bug mentioned in the 2023 competition, which resulted in approximately:

Source	Number of samples
Xeno Canto	7,376
Previous Competitions	90
These samples were not present in the current year's dataset but had appropriate primary_label labels.
Interestingly, using parsed data from XC and iNat did not improve my models but gave a minor performance boost to @vialactea.
That’s all (folks) regarding the main training data.

5 seconds of fame
We trained our models on 5s and randomly selected segments. Based on experience from last year, we initially tried three approaches to picking 5s segments: random 5s from the whole audio, random 5s from the first 7s, and random 5s from the first or last 7s. The reasoning for the last two approaches is that often the recorder starts the audio when the animal is vocalizing and stops it when the animal stops vocalizing. That would help the model avoid false positives. The 7s was just to add some diversity.

In initial tests, the last approach provided better results. However, some species had just a few audios—as few as 2 in some cases—and we were concerned about overfitting. Therefore, we decided to inspect the audios for those species to manually identify sections of vocalization. The following charts illustrate three of the cases we found:

Vocalization with alien speech. By alien speech, we mean speech explaining the recording and with no traces of vocalization or even the same background noise as in the section in which there is vocalization. In the chart, the alien speech occurs both before and after the vocalization.

Speech overlapping with animal vocalization. In this case, the voice can be understood as background noise. In the chart, this occurs between seconds 57 and 142. The final 8 seconds is alien speech.

Vocalization with periods of silence, i.e., when the animal is not vocalizing.

Avoiding false positives must be a good thing, right? Well, no! Skipping random 5s periods in which there is no vocalization actually reduced LB. Why? Were we in over our heads trying to identify vocalization? Was there something in the audio that we were unable to detect but the models could? Maybe those false positives helped generalization by preventing the model from overfitting to the few audios available. We ended up either using the whole audio or avoiding just the sections of alien speech identified either manually or automatically. It seems that fame has more to do with false positives than with our intuition. Perhaps AI is getting too good at imitating biological intelligence, and like people, these models have developed a taste for fake stuff.
Validation Is All You Need but Do Not Have
That is, unfortunately, a true story for all BirdCLEF competitions. We explored a bunch of validation strategies. At the core of each strategy lies stratification by primary_label and grouping by author. After that, we had to figure out how to treat undersampled species. So, what we tried:

Adding at least one sample of each class to each validation fold, even if it introduced a tiiiny data leakage. We also introduced scoring for undersampled species within each fold.
Using the first approach but removing the added sample from the respective training fold.
Adding all undersampled species to the training folds and removing them from the respective validation folds. This allows the model to see more examples of undersampled species and reduces noise in the validation scores of small classes, at the cost of losing validation feedback for those species.
We mostly settled on the first and third strategies.
And of course, the most interesting question: do validation scores correlate with our Public score? After some major improvements, we saw significant positive changes in both metrics, but when tweaking things within ~1% AUC, the correlation was nearly absent.

As you can see in the plot, when we broke the 0.9 Public milestone, the correlation decided to take a rest.

Modelling, let’s be honest, is one of the most useless parts here
We followed the good traditions of previous Bird and Audio competitions and used a Spec → 2D CNN approach. But particularly for this year, backbones really did make the difference! We tried the ConvNeXt family, which showed pretty good results in 2023, but it failed this time—while also introducing very big latency. The ResNeXt family didn’t perform well either. Next encoders have become our favorites:

tf_efficientnetv2_s
eca_nfnet_l0
In some setups, EfficientNet showed better results, while in others, nfnet_l0 carried the day. Additionally, both were a really good match for ensembles.

Regarding classification heads, things were pretty standard: I used a SED head, while @vialactea used a multi-layer perceptron.

Train it to the limit
Common parts for both backbones:

50 epochs
Batch size: 64. We tried larger sizes in the final days of the competition but, unfortunately, didn’t have submissions available to test them.
Scheduler: half of a cosine cycle
Focal + BCE loss
Label smoothing 0.005
Optimizer choice differed between backbones:

eca_nfnet_l0: RAdam with 1e-4 learning rate
tf_efficientnetv2_s: AdamW with 1e-4 learning rate, 1e-8 epsilon, and (0.9, 0.999) betas
I think the most important part here was the balancing strategy. We explored a few and used different ones in the final ensemble:

Balanced
Squared
sample_weights = (
    all_primary_labels.value_counts() / 
    all_primary_labels.value_counts().sum()
)  ** (-0.5)
Upsampling: Repeating samples from the most undersampled classes until each reaches a predefined count (e.g., all classes with fewer than 100 samples are upsampled to have 100).
We ended up using both the Balanced strategy and a combination of Balanced + Upsampling.

And now we come to the first ingredient of our secret sauce—pretraining!

Let me tell you a story about …. Pretraining
From the very first experiments, @vialactea had very good results with fine-tuning from our last year’s pretrained backbones. The general concept is pretty simple:

Download a massive Xeno-Canto dataset, excluding recordings with this year’s species to avoid data leakage
Filter out undersampled species to avoid overcomplicating the classification problem during pretraining—this results in approximately 7,400–7,800 species
Train, train, train!
Picking the right checkpoint is a separate kind of art. We tried using the last, the best, and the average of the 3 best. The last two approaches worked best.
Use only the pretrained backbone and discard the classification head.
Initially, I tried pretraining only on past competition data—and failed. That initialization made things worse. Then I adopted @vialactea approach and confirmed it was a killer feature: scores jumped from 0.83–0.84 to 0.86–0.87.

After the success of last year’s pretrained models, we decided to train new ones on a fresh massive Xeno-Canto snapshot. Unfortunately, it didn’t go as well: they performed comparably or slightly worse than the 2024 pretrains. Still, one eca_nfnet_l0 checkpoint showed promise and was selected for the final ensemble.

After pretraining, we fine-tuned on our main datasets without using sophisticated techniques like differential learning rates for backbone and head.

Augmentations
Here I completely refer to the 2023 write-up, as we used exactly the same setup. We tried turning off RandomFiltering and/or SpecAug, which led to slightly better CV scores but worse LB. So the decision was made: don’t make life easier for the model—robots should work!

Release the Kraken — Pseudo Time!
We used the following algorithm:

Predict all train_soundscapes using our best ensemble in “submission” format (5-second windows).
Select only confident segments—we kept only those with a maximum probability greater than 0.5.
Use soft labels— do not apply thresholding. This can be considered a form of ensemble distillation into a single model, while also adapting to the noise and target distribution of the soundscapes.
Important: Trim low probabilities to zero—we set all probabilities below 0.1 to zero. This is a key step for removing noisy, unconfident labels.
The pseudo-code looks something like this:

…
# primary_label_prob contains max prob across 5 second chunk
soundscape_df = soundscape_df[soundscape_df["primary_label_prob"] > 0.5]
soundscape_df[scored_species <0.1] = 0
…
The next step was sampling correctly. To avoid ruining (further) the already skewed data distribution and to better control the amount of sampled pseudo labels, we used the following strategy:

Sample according to the original sampling strategy.
Check whether the class from the sample exists in soundscape_df. If it does, with some probability (0.4 in our case), select it instead of the current sample and replace the hard label with the pseudo soft label vector.
The first pseudo iteration improved our scores from 0.86–0.87 to 0.89–0.895. But as mentioned—only the first…

The next obvious step? STACK MORE PSEUDO ITERATIONS! So, we predicted using models trained on the first pseudo-iteration samples, this time using the soundscapes that didn’t pass the max-probability threshold. This boosted scores to the 0.90–0.91 range.

Unfortunately, a third iteration didn’t lead to noticeable gains, so we changed the strategy a bit. The main issue was that we couldn’t re-predict soundscapes from the previous iteration due to data leakage. But what if we split the soundscapes into folds? Now we could predict them with the “pseudo-trained” models in OOF (out-of-fold) mode. Sadly, repeating this approach twice didn’t boost scores beyond the second iteration. However, it gave us pseudo-labels generated in a slightly different way—which we then combined with previous pseudo-labels to form a crazy pseudo mix. A tf_efficientnetv2_s-based model trained on this mix became our top solo model, scoring 0.917 Public and 0.91 Private.

Pseudo iteration	Number of selected files
1	4430
2	1483
3	1437
Add a bit of postprocessing
Contrary to our 2023 solution, this year we decided to severely overfit. Not really, of course—but we did select checkpoints based on the best validation ROC AUC. We tried using the last and the average of the 3 best checkpoints, but those usually performed worse on the leaderboard. Fortunately, with fewer files to predict, we didn’t worry too much and simply submitted all 5 fold models per experiment.

We used a simple post-processing step: for each audio file, we multiplied all chunk-level predictions by the top probability for each bird class in that file. This boosts consistently strong predictions and suppresses weaker ones, without affecting the within-file ranking (since all chunks are scaled equally). However, it changes the global ranking across files — e.g., a 0.2 chunk in a strong file (mean = 0.9) becomes 0.18, while in a weak file (mean = 0.4) it becomes 0.08. This consistently improved results, typically by 0.01 for weaker models and around 0.005 for stronger ones. Code for the post-processing function:

def postprocessing(input_df, top=1):
    only_probs = input_df.iloc[:, 1:].values
    N, F = only_probs.shape
    only_probs = only_probs.reshape((N//12, 12, F))
    mean_ = np.mean(np.sort(only_probs, axis=1)[:, -top:], axis=1, keepdims=True)
    only_probs *= mean_
    input_df.iloc[:, 1:] = only_probs.reshape((N, F))
    return input_df
We predicted each 5-second chunk independently during inference. We also tried the overlapping window approach used by the previous TOP 4 team (let’s follow their naming and call it TTA). It performed well on the Public LB, boosting the best experiment’s score from 0.917 to 0.922 on Public and from 0.91 to 0.918 on Private. Unfortunately, it did not perform well with postprocessing on Public

And finally …
Our final submission contained 3 models:

tf_efficientnetv2_s trained on my pipeline with 2 pseudo iterations from the 2024 pretrained checkpoint. Trained on validation split #3 and with a Balanced sampling strategy.
eca_nfnet_l0 trained on my pipeline with 3 pseudo iterations from the 2025 pretrained checkpoint. Trained on validation split #1 and with a Balanced + Upsampling sampling strategy.
tf_efficientnetv2_s trained using @vialactea pipeline with 1 pseudo iteration from the 2024 pretrained checkpoint. Trained on validation split #3 and with a Balanced sampling strategy.
All of this was backed by post-processing on top and without TTA prediction.

A Bit of Failed Stuff
Training on soft labels of the main training data. This seems like a super logical idea, but with deep learning, as usual—you never know what will work.
Using pretrained weights from the latest Xeno Canto snippet
Using additional iNaturalist or XC data
Additional augmentations, like Time Flip
Write-up Speedrun
The strongest ones will cope with our long read, but for others—a short ablation table.

Improvement	Public Score
Baseline	0.83-0.84
+ Pretrain	0.86-0.87
+ Pseudo iteration 1	0.89-0.895
+ Pseudo iteration 2-3	0.9-0.91
+ TTA	0.922
Postprocessing	+0.005-0.01
Closing words
I hope you haven’t fallen asleep while reading. First, I want to congratulate @vialactea on achieving Grandmaster ranking! It was an honour to participate in this competition with you!

I want to thank the entire Kaggle community and congratulate all participants and winners. Special thanks to the Cornell Lab of Ornithology, LifeCLEF, Google Research, Xeno-canto, @stefankahl, @tomdenton, @holgerklinck, and @avocadomastermind. All of you were super active in discussions, shared datasets and interesting materials, answered all questions, and, of course, prepared such a cool competition!

Resources
Inference Kernel : https://www.kaggle.com/code/vladimirsydor/bird-clef-2025-ensemble-v2-final-final?scriptVersionId=244942051
Short Inference Kernel: https://www.kaggle.com/code/vladimirsydor/bird-clef-2025-minimul-inference?scriptVersionId=245080754
GitHub : https://github.com/VSydorskyy/BirdCLEF_2025_2nd_place
Paper : https://ceur-ws.org/Vol-4038/paper_256.pdf

Project Details
Authors
vialactea
vialactea

Volodymyr
vladimirsydor


Share
7
4
9 Comments
Hotness
vialactea
Posted 6 months ago

2nd in this Competition

Thanks, Volodymyr. It was a real pleasure teaming up with you again. You're not only exceptionally bright, but also hard-working, thoughtful, and consistently positive. It was an honor to work with you.


Reply
Qsj
Posted 5 months ago

congratulate！


Reply
vialactea
Posted 5 months ago

2nd in this Competition

Thanks @qiaoshiji.


Reply
Volodymyr Pivoshenko 🇺🇦
Posted 6 months ago

Thanks Volodymyr! It was a very interesting read with some cool ideas!


Reply
Ayça Güler
Posted 5 months ago

Thanks for your informative sharing. Is it possible to construct this methodology on Kaggle kernel without any additional hardware resource?


Reply
Yan Zhang
Posted 5 months ago

337th in this Competition

congratulate！


Reply
Sarah Arshad
Posted 5 months ago

Thanks Man!
It was a very interesting read with some cool ideas…


Reply
Abhisek Dash
Posted 5 months ago

1809th in this Competition

Hello Volodymyr and Vialactea. Congratulations on your medal and rank. Thanks for the clear and concise writeup.

I am trying to replicate your results in Pseudo Labeling as that was something that I did not try.
Here is what I have done till now:

Model: Efficientnet B0-Training from scratch
Oversample train split with less than 100 samples/species to 100
Use class weights
Train augmentation: Frequency and Time Masking of Spectrograms
Ensemble of 2 models trained on different splits

The validation AUC is around 0.82 and train AUC around 0.9 for both models

However while using the models to generate Pseudo labels on train soundscapes, I noticed that none of my predictions are above 0.5.
Any tips to nudge me in the right direction?


Reply
vialactea
Posted 5 months ago

2nd in this Competition

That didn’t happen to us. Several factors affected how many hits we had (5-second segments with probability > 0.5). Two examples are the quality of the model (better models tend to get more hits) and the training data (e.g., including rare species in the training set tended to generate more hits for those species). However, we always got thousands of hits.

My guess is that either you have a bug or your models need further improvement. The validation AUC depends on the validation strategy, and you didn’t specify yours. We tried many different approaches, and all of them produced a validation AUC above 0.90 — in most cases, well above.

In our case, using larger encoders and pretraining them helped improve performance. Other factors that affected the quality of our models included the loss function, optimizer, learning rate, and so on.


Reply
Abhisek Dash
Posted 5 months ago

1809th in this Competition

I am using groupshufflesplit with 30% validation. Also my models are not confident at all. Most max probabilities are in the range of 0.1 to 0.2


Reply

3 more replies
Profile picture for Volodymyr
Profile picture for Abhisek Dash
Overvalue Awareness
Posted 5 months ago

seems like a super logical idea, but with deep learning, as usual—you never know what will work

Noted.


Reply
Skip to
content
Kaggle

Create
Search

Home

Competitions

Datasets

Models

Benchmarks

Game Arena

Code

Discussions

Learn

More


View Active Events

Skip to
content
Kaggle

Sign In

Register
3rd Place Solution

BirdCLEF+ 2025

Solution Writeup · 3rd place · Jun 7, 2025

BirdCLEF2025 Competition Summary
Acknowledgements
Thank you to the organizers for hosting such an amazing and well-structured competition. I also appreciate the inspiring public notebooks (e.g., from @salmanahmedtamu) and excellent solutions from previous years.

Congratulations to all the top teams—especially to my teammate for achieving Grandmaster status! I'm sincerely grateful for his invaluable support throughout this competition.

The details of our solution is listed below:

1. Training Data and Validation Strategy
Like many participants, we used the full BirdCLEF 2025 dataset for training. However, to achieve more stable cv-lb results, we expanded our training set by including 102 additional categories from the BirdCLEF 2023 dataset.

Specifically, we used 80% of the BirdCLEF2023 + 100% of the BirdCLEF2025 data for training, and the remaining 20% of BirdCLEF2023 for validation. This strategy provided a rough indication of model convergence. While the CV-LB correlation was still not perfectly stable, it was much more consistent compared to our earlier attempts.

Additionally, we collected extra data from the following sources:

Xeno-Canto
iNaturalist
We also cleaned parts of the CSA dataset using a public notebook to remove human voices and manually filtered the remaining samples for quality assurance.

Moreover, our training data included train_soundscape audios with pseudo labels generated by an ensemble of models trained on the dataset described above.

2. Models
Our final system combined both CNN-based and SED-based models, utilizing the following backbones:

tf_efficientnet_b0_ns
tf_efficientnetv2_b3
tf_efficientnetv2_s.in21k_ft_in1k
mnasnet_100
spnasnet_100
We used two sets of Mel spectrograms with the following parameters (The only variation between the two sets lies in
the n_mels parameter, which was set to either 128 or 96):

mel_spec_params = {
 "sample_rate": 32000,
 "n_mels": 128 or 96,
 "f_min": 0,
 "f_max": 16000,
 "n_fft": 2048,
 "hop_length": 512,
 "normalized": True,
 "center": True,
 "pad_mode": "constant",
 "norm": "slaney",
 "mel_scale": "htk",
}
3. Training Strategy
Sampling:
Random sampling outperformed fixed-first-5s and RMS-based approaches.
Augmentations:
We followed augmentation strategies from prior competitions (cutmix, mixup, sumix) and added additional human
voice noise as background noise for robustness.
Loss Function:
Both CNN and SED models were trained with Focal Binary Cross Entropy (Focal BCE) loss.
Model Soup:
We applied model weight averaging across multiple checkpoints to stabilize final predictions.
4. Final Submission
We applied a rank-aware post-processing strategy inspired by:
Post-processing with power adjustment
This helped adjust low-confidence predictions based on ranking.

All models were exported to ONNX format for fast inference.
Ensemble of 20 models: 10 CNN models and 10 SED models (5 backbones x 2 sets of mel params). This submission can reach private lb=0.927. However, our best lb is achieved by 6 models each, without mnasnet and spnasnet backbones.
Project Details
Authors
shanzhong8
shanzhong8

Leon
leonshangguan


Share
2
4
1
15 Comments
1 appreciation comment
Hotness
MYSO
Posted 6 months ago

5th in this Competition

Congrats! Your solution was really educational and inspiring.
Glad to hear the notebook I shared was useful, even just a little!


Reply
shanzhong8
Topic Author
Posted 6 months ago

3rd in this Competition

Congrats on your cash gold medal as well—excellent and solid work!


Reply
RB
Posted 6 months ago

73rd in this Competition

Well done and Congratulations !

we expanded our training set by including 102 additional categories from the BirdCLEF 2023 dataset.

It seems Birdclef 2023 had 264 species, how did you choose 102 ?


Reply
shanzhong8
Topic Author
Posted 6 months ago

3rd in this Competition

Thank you for the question! We selected the 102 classes based on the number of samples, choosing those specoes with more than 50 (or 40) instances. Constructing a more balanced dataset allows us to better investigate the generalization ability of the model during training.


Reply
Overvalue Awareness
Posted 6 months ago

Congrats…
Would you mind sharing the training code?


Reply
Corentin Lobet
Posted 6 months ago

1672nd in this Competition

Congratulations, you've done some great engineering there 👏


Reply
Yassine Alouini
Posted 6 months ago

372nd in this Competition

Congratulations on the win!
How much did the onnx exported versions speed up the inference? Have you measuere it?


Reply
shanzhong8
Topic Author
Posted 6 months ago

3rd in this Competition

Thanks for the question. That's approximately 2 to 3 times faster in our case.


Reply
Ty-Yuki
Posted 6 months ago

754th in this Competition

Congrats!
Could you elaborate on why random sampling outperformed other approaches for training?


Reply
shanzhong8
Topic Author
Posted 6 months ago

3rd in this Competition

If high-quality pseudo labels are available, it is essential to make full use of the data. When certain training samples appear to be noisy, they can be transformed into improved pseudo labels to better guide the model’s learning process， rather than dropped them by the RMS/first 5s approch. I mean, you may get some LB gain from these approches at an early stage, but to further boosting your score, you should figure out how to make full use of the data.


Reply
1
thacrobatheskis
Posted 6 months ago

710th in this Competition

Thanks for sharing! Did you use 20% of 2023 for early stopping as well? Otherwise, how did you determine how many epochs to train?


Reply
shanzhong8
Topic Author
Posted 6 months ago

3rd in this Competition

Yes correct.


Reply
Aayush Kumar Singha
Posted 6 months ago

313th in this Competition

Congratulations for the 3rd position!! An amazing feat!!

I do have a question.
When I was analysing the past datasets, here are the common labels count with 2025 I found.


How did you use 80% + 20% (validation) of the 2023 dataset?


Reply
shanzhong8
Topic Author
Posted 6 months ago

3rd in this Competition

Thanks for your visualization and good question. In this competition, the model is trained on the entire 2025 dataset. Additionally, one fold (e.g., fold 0) from the 2023 dataset is incorporated, where 20% of the fold is used as the validation set and the remaining 80% is merged into the training set.


Reply

Appreciation (1)
dedq
Posted 6 months ago

So cool for sharing, thank shanzhong8!

Skip to
content
Kaggle

Create
Search

Home

Competitions

Datasets

Models

Benchmarks

Game Arena

Code

Discussions

Learn

More


View Active Events

Skip to
content
Kaggle

Sign In

Register
4th place solution

BirdCLEF+ 2025

Solution Writeup · 4th place · Jun 11, 2025

First of all, this is my first time to reach the prize on kaggle, much thanks to the organizers and all the participants.

My solution is a SED solution inspired by birdclef2023 2nd place solution, combined with a custom soft AUC loss and semi-supervised learning.

Main concept: soft AUC loss

I always thought that the best loss should be the metric itself, so I tried to find an AUC loss and found:

class AUCLoss(nn.Module):
    def __init__(self, margin=1.0, pos_weight=1.0, neg_weight=1.0):
        super().__init__()
        self.margin = margin
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight

    def forward(self, preds, labels, sample_weights=None):
        pos_preds = preds[labels == 1]
        neg_preds = preds[labels == 0]

        if len(pos_preds) == 0 or len(neg_preds) == 0:
            return torch.tensor(0.0, device=preds.device)

        if sample_weights is not None:
            sample_weights = torch.stack([sample_weights]*labels.shape[1], dim=1)
            pos_weights = sample_weights[labels == 1]  # [N_pos]
            neg_weights = sample_weights[labels == 0]  # [N_neg]
        else:
            pos_weights = torch.ones_like(pos_preds) * self.pos_weight
            neg_weights = torch.ones_like(neg_preds) * self.neg_weight

        diff = pos_preds.unsqueeze(1) - neg_preds.unsqueeze(0)  # [N_pos, N_neg]
        loss_matrix = torch.log(1 + torch.exp(-diff * self.margin))  # [N_pos, N_neg]

        weighted_loss = loss_matrix * pos_weights.unsqueeze(1) * neg_weights.unsqueeze(0)

        return weighted_loss.mean()
This AUC loss seems to be very resistant to overfitting. In all experiments, the cv scores of the models trained with the cross entropy loss were significantly better than those trained with the soft AUC loss, but the lb scores were significantly worse.
There is a problem with this AUC loss: it does not support soft labels like cross entropy loss. For both knowledge distillation and semi-supervised learning, I need a loss that supports soft labels, so I made some changes to the above AUC loss:

class SoftAUCLoss(nn.Module):
    def __init__(self, margin=1.0, pos_weight=1.0, neg_weight=1.0):
        super().__init__()
        self.margin = margin
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight

def forward(self, preds, labels, sample_weights=None):
        pos_preds = preds[labels>0.5]
        neg_preds = preds[labels<0.5]
        pos_labels = labels[labels>0.5]
        neg_labels = labels[labels<0.5]

        if len(pos_preds) == 0 or len(neg_preds) == 0:
            return torch.tensor(0.0, device=preds.device)

        pos_weights = torch.ones_like(pos_preds) * self.pos_weight * (pos_labels-0.5)
        neg_weights = torch.ones_like(neg_preds) * self.neg_weight * (0.5-neg_labels)
        if sample_weights is not None:
            sample_weights = torch.stack([sample_weights]*labels.shape[1], dim=1)
            pos_weights = pos_weights * sample_weights
            neg_weights = neg_weights * sample_weights

        diff = pos_preds.unsqueeze(1) - neg_preds.unsqueeze(0)  # [N_pos, N_neg]
        loss_matrix = torch.log(1 + torch.exp(-diff * self.margin))  # [N_pos, N_neg]

        weighted_loss = loss_matrix * pos_weights.unsqueeze(1) * neg_weights.unsqueeze(0)

        return weighted_loss.mean()
This soft AUC loss + semi-supervised learning improved my single tf_efficientnetv2_b0 model's lb score from 0.850 to 0.901. More importantly, my improvement from 11th to 4th on private lb is probably due to the use of this loss.

Other things that helped
Semi-supervised learning. The labeling model is 10 SED models of efficientnet_b0-b4, efficientnetv2_b0-b3 and efficientnetv2_s trained wiht first 10s audio data.
Smaller hop_length (64) and larger n_mels (256).
Audio mixup augmentation (which is to add two audios as the new audio and take the maximum value of their labels as the new label). This augmentation did not directly improve my model, but in order to increase the diversity of the final solution, I still added this to the training of some models.

Things that didn't help or make it worse
Any kind of pretraining.
Knowledge distillation.
Models other than efficientnet.
Data normalizitions other than 2D batch normalizition.

Final models
16models of efficientnet_lite0-4, efficientnet_b2-3, efficientnetv2_b2-3 and efficientnetv2_s. 17-25 epochs, learning rate 5e-4. 3 types of mel spectrogram parameters. 2 types of data augmentation. First 10s data and random 10s data.

Code
https://github.com/dylanliu2/BirdCLEF2025-4th-place-solution

Project Details
Author
Dylan Liu
dylanliuofficial


Share
1
3 Comments
Hotness
Tom Denton
Competition Host
Posted 5 months ago

Cool, thanks!

BTW, the loss matrix computation can be more numerically stable by using the softplus operation - which computes log(1+exp(x)) using numerically stable ops - otherwise you might suffer from under/overflow issues.


Reply
I2nfinit3y
Posted 5 months ago

9th in this Competition

Nice sharing ! By the way, you can use ``` to edit your code block. Just like:

class SoftAUCLoss(nn.Module):
    def init(self, margin=1.0, pos_weight=1.0, neg_weight=1.0):
        super().init()
        self.margin = margin
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight

    def forward(self, preds, labels, sample_weights=None):
        pos_preds = preds[labels>0.5]
        neg_preds = preds[labels<0.5] pos_labels = labels[labels>0.5]
        neg_labels = labels[labels<0.5]
        if len(pos_preds) == 0 or len(neg_preds) == 0:
            return torch.tensor(0.0, device=preds.device)

        pos_weights = torch.ones_like(pos_preds) * self.pos_weight * (pos_labels-0.5)
        neg_weights = torch.ones_like(neg_preds) * self.neg_weight * (0.5-neg_labels)
        if sample_weights is not None:
            sample_weights = torch.stack([sample_weights]*labels.shape[1], dim=1)
            pos_weights = pos_weights * sample_weights
            neg_weights = neg_weights * sample_weights

        diff = pos_preds.unsqueeze(1) - neg_preds.unsqueeze(0)  # [N_pos, N_neg]
        loss_matrix = torch.log(1 + torch.exp(-diff * self.margin))  # [N_pos, N_neg]

        weighted_loss = loss_matrix * pos_weights.unsqueeze(1) * neg_weights.unsqueeze(0)

        return weighted_loss.mean()

Reply
Dylan Liu
Topic Author
Posted 5 months ago

4th in this Competition

Ok, thanks


Reply
Skip to
content
Kaggle

Create
Search

Home

Competitions

Datasets

Models

Benchmarks

Game Arena

Code

Discussions

Learn

More


View Active Events

Skip to
content
Kaggle

Sign In

Register
5th place solution: Self-Distillation is All You Need

BirdCLEF+ 2025

Solution Writeup · 5th place · Jun 16, 2025

We would like to thank Kaggle, the organizers, our teammates, and all participants. It was a great experience to take part in this competition. Below is a brief overview of our solution.

Data
We used only the 2025 dataset.
First, we used Silero VAD to detect audio files that contain human voices from the train_audio set. Next, using a Streamlit tool developed by @zuoliao11 , we manually listened to those files and removed the segments with human voices.

For underrepresented classes (n < ~30), we manually selected segments that contained bird calls.
For cleaned files, we used the first 60 seconds; for the others, we used the first 30 seconds. To balance the dataset, we duplicated files in classes with fewer than 20 samples.

Model
We used a Sound Event Detection (SED) model.
Backbones:
• 4x tf_efficientnetv2_s
• 3x tf_efficientnetv2_b3
• 4x tf_efficient_b3_ns
• 2x tf_efficient_b0_ns

Training
We trained our models in three stages.


Input features:
• Random 10-second segments
• Mel spectrogram:

  sample_rate: 32000 
  mel_bins: 192 
  fmin: 20 
  fmax: 15000 
  window_size: 2048 
  hop_size: 768
We convert the mel-spectrogram to a logarithmic scale by computing log(melspec+1e-6).
• Augmentations:
　　• Resampling
　　• Gain
　　• FilterAugment
　　• FrequencyMasking, TimeMasking
　　• Sumix on mel domain

• Optimizer: Adam + Cosine Annealing with warmup
• Loss: FocalLoss (gamma=2)
• Epochs: 10
• Target labels: Both primary and secondary labels

1st Stage:
We trained 5-fold models using only train_audio.

2nd Stage - Self-distillation with train_audio only:
While listening to the audio as described in the Data section, we found that many bird calls were present in the training data even though they were not labeled. This was expected, as the recorders were mainly focused on their target species, so other bird calls were often left unlabeled.
From this observation, we believed that the core challenge of the competition was accurately assigning secondary labels. To address this, we used self-distillation to enrich train_audio with more secondary labels. We used predictions from a model trained in the 1st stage as teacher labels and mixed them with the original labels. The teacher model’s predictions may have included true secondary labels that were missing from the original annotations.
We repeated self-distillation 4–5 times. From the 2nd round onward, we used the previously distilled model as the new teacher in an iterative manner. The model’s weights are re-initialized each time. This approach closely resembles the method proposed in this paper.

3rd Stage - Self-distillation with train_audio + train_soundscapes:
We added data from train_soundscapes to the training set and continued self-distillation two more times. We mixed train_audio and train_soundscapes at a 1:1 ratio in each batch (no folds). We further trained several models using different random seeds.

LB Score (Public) 1st ~3rd stage（Ensemble of 5 models）
Model	Stage 1	Stage 2 (Distill x2)	Distill x4	Distill x5	Stage 3 (Distill x1)	Distill x2
tf_efficientnetv2_s	0.839	0.863	0.880	0.884	0.915	0.921
tf_efficientnetv2_b3	0.842	N/A	0.872	-	N/A	0.918
tf_efficient_b3_ns	N/A	N/A	N/A	-	N/A	0.921
tf_efficient_b0_ns	0.836	0.871	0.879	0.883	0.905	0.912
*N/A = not submitted to LB

The following figure shows the results of self-distillation across various models.


Inference
We divided the stage 3 models into two groups, assigning different random seeds to each group when possible.

Model Group A:
4x tf_efficientnetv2_s (seed= 0, 1, 2, 3)
3x tf_efficientnetv2_b3 (seed= 2, 3, 4)
4x tf_efficient_b3_ns (seed= 0, 1, 2, 3)
2x tf_efficient_b0_ns (seed= 0, 1)

Model Group B:
4x tf_efficientnetv2_s (seed= 1, 2, 3, 4)
3x tf_efficientnetv2_b3 (seed=0, 1, 2)
4x tf_efficient_b3_ns (seed= 0, 1, 2, 3) *By mistakes, we ended up using the same seed as Group A.
2x tf_efficient_b0_ns (seed= 2, 3)

Post-Proccesing/TTA
Inference is done with 2.5-second overlap.
Scores are weighted and combined (similar to the 4th place solution from last year).
Alpha = 0.5


Smoothing:
We applied smoothing using neighboring frames with a window of [0.1, 0.8, 0.1].

Power Adjustment for Low-Ranked Classes:
The post-processing method shared in our public notebook improved the LB score, but we eventually decided not to use it due to the risk of overfitting.

Speed-up:
• OpenVINO
• Concurrent.futures.ThreadPoolExecutor

Final Ensemble LB Scores
Setting	Raw Score (Group A only)	2.5-second overlap	Smoothing + overlap
Public LB	0.919	0.928	0.928
Private LB	0.917	0.924	0.924
What didn’t work
CNN-based models.
1D models.
Too many data augmentations.

Training, Inference Notebooks & Model Dataset
Training code
Github
Models
PyTorch
OpenVINO
Inference Notebooks
Convert PyTorch models to OpenVINO
Inference
Project Details
Authors
Zuoli
zuoliao11

MYSO
myso1987

youknow
yuukiyuno


Share
5
22 Comments
2 appreciation comments
Hotness
vineet dairashri
Posted 5 months ago

230th in this Competition

Hello MYSO,
Thanks for sharing this incredible project insights. Can you also share little bit about the training of the model that you didn’t use which is low power adjustment ecanet model?

Thanks alot!

Best,
Vineet


Reply
MYSO
Topic Author
Posted 5 months ago

5th in this Competition

The EcaNet model in the public notebook was trained with essentially the same pipeline as our other models. I’ve posted the full training code on GitHub—please feel free to check it out and put it to use!


Reply
Overvalue Awareness
Posted 6 months ago

Congrats!
Would you explain how many hours it take to train 1st, 2nd and 3rd stage, respectively?
What GPU did you use? Do you own it or rent?
If own how much did you buy it? Did you put the GPU inside your computer by yourself or from maker?
If rent where do you rent? How much it cost?

Thank you.


Reply
MYSO
Topic Author
Posted 6 months ago

5th in this Competition

We used an RTX 3090. Depending on the model architecture, one epoch took approximately 5–10 minutes in stage 1, 7–15 minutes in stage 2, and 12–18 minutes in stage 3.
The GPU was included in a pre-built PC from the manufacturer.
Since GPU prices fluctuate a lot, we recommend checking with the manufacturer for current pricing.


Reply
pensukesan
Posted 6 months ago

292nd in this Competition

Thank you for sharing your solution.
I have two questions regarding self-distillation:

In the first stage, the model is trained using 5-fold cross-validation. In the second and third stages, are you also training using 5-fold models? If so, how are the teacher models from the first stage assigned to each fold?

How are the pseudo labels and original labels mixed?


Reply
MYSO
Topic Author
Posted 6 months ago

5th in this Competition

In the second stage, we trained using 5-fold cross-validation. In the third stage, we trained without folds, using different random seeds instead.

For self-distillation, we used the five models from the previous stage and averaged their predictions to create the teacher labels.
As for mixing pseudo labels with the original labels, we generally used the following approach:

alpha=0.7
pseudo_labels = pseudo_labels * (pseudo_labels > 0.3) + pseudo_labels**2
pseudo_labels = torch.clamp(pseudo_labels, min=-0.0, max=1.0)
labels = alpha*pseudo_labels + (1-alpha)*labels

Reply
1
Ty-Yuki
Posted 6 months ago

754th in this Competition

Thanks for sharing your 5th place solution, especially the iterative self-distillation!

I'm curious about the manual audio cleaning. How much data did you manually process, and how critical do you think this step was for your final score?


Reply
MYSO
Topic Author
Posted 6 months ago

5th in this Competition

Thank you! We manually cleaned about 2,000 audio files.
Sorry, we haven't specifically measured how much this step affected the final score.
But doing it early allowed us to focus more on model building and self-distillation, so we believe it was an important step.


Reply
1
EMMANUEL OKELLO
Posted 6 months ago

1723rd in this Competition

Congrats.on making it 5th and thanks for sharing your solution. I have learned a lot from it and hope to try it out on my other model trainings.


Reply
MYSO
Topic Author
Posted 6 months ago

5th in this Competition

Thanks a lot!
Happy to hear it was helpful. Good luck with your next training — hope it goes well!


Reply
1
Benchan
Posted 6 months ago

1353rd in this Competition

Congratulations on achieving fifth place, and thank you sincerely for sharing the details of your work.

"While listening to the audio as described in the Data section, we found that many bird calls were present in the training data even though they were not labelled."

I had suspected that many of the provided recordings contained unlabeled secondary bird calls, but your diligent and meticulous manual screening brought this issue to light. I believe this insight significantly contributed to your impressive result.

Your openness in sharing your approach has been immensely instructive. I am truly grateful.


Reply
MYSO
Topic Author
Posted 6 months ago

5th in this Competition

I truly appreciate your encouragement.


Reply
Aman Rajesh Kapoor
Posted 6 months ago

567th in this Competition

congrats on the 5th place . I have one question about the self distillation did you use the same model for teacher & student and repeat the process n times , with different data for teacher but with constant data for student or the student data changes as the distillation progresses


Reply
MYSO
Topic Author
Posted 6 months ago

5th in this Competition

Thank you! We used the same model for both teacher and student every time, and the data stayed the same throughout the self-distillation process.


Reply
Aman Rajesh Kapoor
Posted 6 months ago

567th in this Competition

thanks for the reply 😄


Reply
1
Rose Beltran
Posted 6 months ago

Manually listening to and editing the files shows incredible dedication. Nice to see that the effort was rewarded with insight about self-distillation, leading to this exceptional result. Thank you for sharing your solution and congratulations!


Reply
MYSO
Topic Author
Posted 6 months ago

5th in this Competition

Thanks a lot! We really appreciate your comment.


Reply
thacrobatheskis
Posted 6 months ago

710th in this Competition

Thanks for sharing! The distillation technique you used sounds quite interesting. Did you use the fine or coarse grain outputs from the SED teacher model to train the student? Also, was the teacher given the augmented training data, or only the student? I’m looking forward to checking out your train notebook when it’s available!


Reply
MYSO
Topic Author
Posted 6 months ago

5th in this Competition

We also used the augmented data for the teacher model, which produced coarse-grain outputs.


Reply
Tarun Pahade222
Posted 6 months ago

Really intresting


Reply
1
This comment has been deleted.


Appreciation (2)
Athar Sayed
Posted 6 months ago

564th in this Competition

Amazing solution thanks for sharing

Sarah Arshad
Posted 6 months ago

thanks alot….

Skip to
content
Kaggle

Create
Search

Home

Competitions

Datasets

Models

Benchmarks

Game Arena

Code

Discussions

Learn

More


View Active Events

Skip to
content
Kaggle

Sign In

Register
6th place solution

BirdCLEF+ 2025

Solution Writeup · 6th place

6th place solution
Thank you to the organizers for hosting this excellent competition, and also to all the participants who shared valuable insights throughout the competition period.

I was greatly inspired by the top solutions from past competitions. Thank you to everyone who shared their approaches.
In this post, I will mainly highlight the differences in my approach.

Model / Loss
I used a SED-style model like the one below:

class AttBlockV2(nn.Module):
    def __init__(self, in_features: int, out_features: int, activation="sigmoid"):
        ...
        self.activation = activation
        ...

    def forward(self, x):
        norm_att = torch.softmax(torch.tanh(self.att(x)), dim=-1)
        cla = self.nonlinear_transform(self.cla(x))
        x = (norm_att * cla).sum(2)
        return x, norm_att, cla

   def nonlinear_transform(self, x):
        if self.activation == "linear":
            return x
        elif self.activation == "sigmoid":
            return torch.sigmoid(x)

class BirdModel(nn.Module):
    def __init__(self, cfg, pretrained: bool = True):
        ...
        self.encoder = timm.create_model(
            cfg.backbone,
            pretrained=cfg.pretrained,
            num_classes=0,
            global_pool="",
            in_chans=cfg.in_chans,
            drop_path_rate=0.2,
            drop_rate=0.5,
        )
        ...
        self.att_block = AttBlockV2(in_features, self.num_classes, activation="sigmoid")
        ...

    def forward(self, x, y=None):
        ...
        clipwise_output, norm_att, segmentwise_output = self.att_block(x)
        segmentwise_logit = self.att_block.cla(x).transpose(1, 2)
        if self.training:
            return clipwise_output, segmentwise_logit.max(1)[0], y
        else:
            return clipwise_output, segmentwise_logit.max(1)[0]
During training, I applied nn.BCEWithLogitsLoss to both clipwise_output and segmentwise_logit.max(1)[0].
Although clipwise_output passes through a sigmoid before loss computation (making BCEWithLogitsLoss technically inappropriate), this setup significantly improved my public score.

When using timm/tf_efficientnet_b3.ns_jft_in1k as the backbone and submitting with clipwise_output (without using train_soundscape), I achieved a public score of 0.900 and a private score of 0.908. At this stage, clipwise_output gave better results than segmentwise_logit.

Pseudo Labeling
I added pseudo labels to train_soundscapes and ran several training cycles with them included as training data.

In the first round, I used only the clipwise_output from a single model (timm/tf_efficientnet_b3.ns_jft_in1k) to generate pseudo labels.

From the second round onwards, I used an ensemble of multiple models' segmentwise_logit.max(1)[0] outputs for pseudo labeling.

Since clipwise_output was trained somewhat unnaturally with BCEWithLogitsLoss, its values were too small and didn't help improve the score when reused in pseudo labeling.

Using segmentwise_logit.max(1)[0] for pseudo labeling led to higher public scores.

Models used for generating pseudo labels included:

timm/tf_efficientnet_b3.ns_jft_in1k

timm/tf_efficientnet_b5.ns_jft_in1k

timm/tf_efficientnetv2_b3.in21k

Finally, I trained the following two models on the combined training data and pseudo labels, and used them for the final submission:

timm/tf_efficientnet_b3.ns_jft_in1k

timm/tf_efficientnetv2_b3.in21k

Score
Public Score: 0.928

Private Score: 0.923

Project Details
Author
Takoi
takoihiraokazu


Share
3 Comments
Hotness
Ty-Yuki
Posted 6 months ago

754th in this Competition

Thank you for sharing your 6th place solution! It's very insightful to see your approach, especially the details on the SED-style model and the pseudo-labeling strategy.

I have a question regarding your loss function: You mentioned applying nn.BCEWithLogitsLoss to clipwise_output even though it passes through a sigmoid before loss computation. Could you elaborate on why this particular setup significantly improved your public score, despite being "technically inappropriate"? Was there any specific observation or intuition that led you to this choice?


Reply
Takoi
Topic Author
Posted 6 months ago

6th in this Competition

Thank you for your question.

I believe this setup might have helped prevent overfitting to the training data. Normally, when the model's output is low, it would take negative values before applying the sigmoid. However, since the output is passed through a sigmoid beforehand, it can no longer take negative values. As a result, when the label is 0, the model is forced to produce very low output values; otherwise, the loss would become large. This tendency to keep the outputs low might have contributed to reducing overfitting.

As for why I used this setup — to be honest, there wasn’t any particular reason. I just tried it out, and it happened to improve my public score, so I continued using it until the end.


Reply
1
Andrew h
Posted 6 months ago

SED-style model, its a great decision


Reply
Skip to
content
Kaggle

Create
Search

Home

Competitions

Datasets

Models

Benchmarks

Game Arena

Code

Discussions

Learn

More


View Active Events

Skip to
content
Kaggle

Sign In

Register
7th place solution

BirdCLEF+ 2025

Solution Writeup · 7th place

I'd like to start by thanking the organizers of BirdCLEF 2025 for hosting this fantastic competition. It was a challenging yet rewarding experience.
Congratulations to all the participants for their hard work and brilliant solutions. I'm excited to share the approach that led to our result.

I also want to give a big thanks to my teammate, rihanpiggy. We've tackled many Kaggle competitions together—sometimes as rivals, sometimes as teammates—and I've learned so much from him along the way. I wouldn't have reached Grandmaster without his hard work and support.

TL;DR
Our solution consists of an ensemble of two model types: SED-style CNNs and 2021 2nd-place style CNNs. As in previous years, combining models trained with different pipelines proved effective for improving our leaderboard score. For the SED model, constructing high-quality pseudo-labels was crucial to achieving strong performance. Small tricks such as post-processing and test-time augmentation also proved consistently effective this year.

CNNs 1: RihanPiggy part
Train dataset
Competition data (manually remove human voice from CSA audio files)
Extra audio files downloaded from Xeno-canto
Models
Blending expert models

SED with tf_efficientnetv2_s_in21k (all species)
SED with hgnetv2_b5.ssld_stage2_ft_in1k (146 aves species)
SED with tf_efficientnetv2_s_in21k (70 aves species which have many training samples)
CNN with hgnetv2_b3.ssld_stage2_ft_in1k (70 major aves species which have many training samples)
SED with hgnetv2_b5.ssld_stage2_ft_in1k (other rare 136 species which have relatively few training samples)
methods which works
Iterative training with pseudo labeling
Inspired by the 3rd place of BirdClef2024, during training, we randomly sample audio clips from train soundscapes and corresponded pseudo label with a probability 50%.
As a result, every train batch contains 50% of train soundscapes and 50% of train audios.
This method works well for all species model and 70 major species model, but not for 146 aves species model and other rare 136 species model.
Iteratively run the cycle of training and pseudo labeling keeps improving the LB. We run this cycle for 4 iterations.
For 70 major aves species model, the pseudo label must be normalized with labels = labels - np.min(labels) to make the approach work.

smoothing postprocess(not used because of the ensemble LB drop)
Smoothing with 2 neighbors using the weight [0.1, 0.8, 0.1] improves both public and private LB for every model.

Extract audio clips from train soundscapes with birdnet
For 146 aves species model and 70 major aves species model, adding audio clips extracted from train soundscapes with birdnet works well.
Birdnet covers 145 aves species, we perform inference on train soundscapes with birdnet and extract audio clips with confidence > 0.1.
This method works well for 146 aves species model and 70 major aves species model, but not for all species model and other rare 136 species model.
This method significantly improves Public LB, but only slightly improves Private LB.

Tricks implemented only on other rare 136 species model
Prevent insecta from mixing up with other species. Insecta only mixup with other insecta species. This trick improves Public LB, but strongly damages Private LB.
0.25 * focal loss. Recovers Private LB damaged by trick 1.
Besides iterative training mentioned above, extracting audio clips from train soundscapes with pseudo labels improves Public LB a lot, but slightly damage Private LB.
After all, baseline shows best Private LB.
Linear model merge
Model merge improves the LB of tf_efficientnetv2_s_in21k (all species, 70 major aves species) model. Merging too much models will damage the LB, 3 models are enough.
For hgnetv2 models, however, model merge will destroy the model.

Model diversity matters
Raw signal model, Simple CNN model increase ensemble LB.

train settings and LB
No	Experiment	architecture	train duration	n_mels	n_fft	fmin	fmax	image_size	Public LB	Private LB
1	all species model baseline	SED v2s	10s	256	2048	0	16000	384	0.866	0.873
2	all species model pseudo iter1	SED v2s	10s	256	2048	0	16000	384	0.894	0.893
3	all species model pseudo iter2	SED v2s	10s	256	2048	0	16000	384	0.898	0.900
4	all species model pseudo merge iter2 iter1 baseline	SED v2s	10s	256	2048	0	16000	384	0.891	0.908
5	all species model pseudo merge iter3 iter2 iter1	SED v2s	10s	256	2048	0	16000	384	0.886	0.903
6	all species model pseudo merge iter4 iter3 iter2	SED v2s	10s	256	2048	0	16000	384	0.894	0.908
No	Experiment	architecture	train duration	n_mels	n_fft	fmin	fmax	image_size	Public LB	Private LB
1	70 major aves species model baseline	SED v2s	10s	256	2048	0	16000	384	0.692	0.695
2	70 major aves species model pseudo iter1	SED v2s	10s	256	2048	0	16000	384	0.696	0.707
3	70 major aves species model pseudo iter2	SED v2s	10s	256	2048	0	16000	384	0.698	0.708
4	70 major aves species model pseudo merge iter2 iter1 baseline	SED v2s	10s	256	2048	0	16000	384	0.699	0.710
5	70 major aves species model pseudo merge iter3 iter2 iter1	SED v2s	10s	256	2048	0	16000	384	0.699	0.710
6	70 major aves species model pseudo merge iter4 iter3 iter2	SED v2s	10s	256	2048	0	16000	384	0.700	0.705
No	Experiment	architecture	train duration	n_mels	n_fft	fmin	fmax	image_size	Public LB	Private LB
1	70 major aves species model baseline	CNN hgnetv2_b3	15s	192	2048	50	14000	288	0.666	0.667
2	70 major aves species model pseudo iter1	CNN hgnetv2_b3	15s	192	2048	50	14000	288	0.687	0.687
3	70 major aves species model pseudo iter2	CNN hgnetv2_b3	15s	192	2048	50	14000	288	0.689	0.685
4	70 major aves species model pseudo iter3	CNN hgnetv2_b3	15s	192	2048	50	14000	288	0.694	0.692
5	70 major aves species model pseudo iter4	CNN hgnetv2_b3	15s	192	2048	50	14000	288	0.688	0.689
No	Experiment	architecture	train duration	n_mels	n_fft	fmin	fmax	image_size	Public LB	Private LB
1	146 aves species model	SED hgnetv2_b3	10s	256	2048	50	14000	288	0.787	0.797
2	other rare 136 species model	SED hgnetv2_b3	10s	256	2048	50	14000	288	0.684	0.662
3	RihanPiggy models ensemble	-	-	-	-	-	-	-	0.916	0.906
Inference
Openvino with async inference queue
Quantize SED models with nncf
What did not work
SED model with train duration 60s to directly catch all the global context in soundscapes.
auxiliary loss of species family classification.
amphibia, mammalia, insecta expert models. Adjusting fmin and fmax to a narrower range for amphibia makes the model even worse.
extract audio clips from train soundscapes with birdvocal.
Separate mixup to different species groups, e.g. aves, insecta, mammalia, amphibia.
Implementing pseudo labeling approach from BirdClef2024 2nd place.
Calculate peaks of each audio and sample audios from peaks. (Somethng like RMS sampling)
Knowledge distillation with logits from birdnet and birdvocal. (Which works for me in BirdClef2024)
CNNs 2: yokuyama part
Training Dataset
Competition data (manually processed to remove human voices from CSA audio files)
Time-series handlabeled competition data (for some rare classes)
Models
I used two CNN models based on the 2021–2nd place solution style. Both models were trained on the same dataset using identical mel-spectrogram settings and backbone. The only difference lies in the length of the audio chunks used during training and inference: one model used 5-second chunks, while the other used 8-second chunks.

The baseline configuration was as follows:

n_mels: 256
n_fft: 2048
fmin: 50
fmax: 14000
image_width: 288
backbone: hgnetv2_b3.ssld_stage1_in22k_in1k
pooling: gem with learnable p
head: linear
loss: bce
optimizer: adamw
max_lr: 1e-3
weight_decay: 1e-3
lr_scheduler linear
epochs: 40
batch_size: 64
augmentation: AddBackgroundNoise, Gain, NoiseInjection, GaussianNoiseSNR, PinkNoiseSNR, SpecAugment, SpecMixup
inference time: ~3min. (single model)
TTA
A simple test-time augmentation was effective: we normalized the audio to a fixed peak volume (0.1) and averaged the model predictions from both the original and normalized mel-spectrograms. This improved robustness to volume variations.

for audio in test_loader:
    audio_norm = 0.1 * audio / audio.abs().amax(dim=1, keepdim=True) # BxL, peak_volume = 0.1
    spec = melspec_transform(audio)
    spec_norm = melspec_transform(audio_norm)
    preds_8s = 0.5 * (model_8s(spec) + model_8s(spec_norm))
Postprocessing
Smoothing with kernel [0.1, 0.3, 1.2, 2.4, 1.2, 0.3, 0.1]
Results
Experiment	private LB	public LB
5s CNN	0.858	0.855
8s CNN	0.851	0.854
5s + 8s + TTA	0.860	0.861
What did not work
Pseudo labeling: Despite significant effort, it did not lead to noticeable performance gains, unlike in SED models.
Auxiliary losses based on taxonomic class and family.
Auxiliary losses based on scale-invariant SNR, SAR, other audio properties…
Alternative front-ends other than mel-spectrograms (e.g., CQT, PECN) were not effective.
Additional Experiments (Not Fully Verified)
Some DINO-pretrained ViT models showed better performance than CNNs.
vit_small_patch14_reg4_dinov2 achieved a promising private LB score in the high 0.88s even in early experiments, outperforming our hgnet. However, due to inference time constraints, it was not included in the final submission.
More thorough human-in-the-loop cleansing and manual label correction had limited impact on the public LB but led to a clear gain of around +0.02 on the private LB.
Ensembles
For the final ensemble, we applied min-max scaling to each model's logits, following the method used by the 11th-place team in the 2024 competition. The scaled logits were then combined using a weighted mean.

We also adopted a soundscape-level postprocessing technique inspired by the 3rd-place team from 2024, which boosted the leaderboard score by ~0.01. The method enhances each soundscape chunk using its maximum logits. A simplified version is shown below:

def postprocess(df, bird_cols, p=0.5):
    # Apply per-soundscape adjustment
    preds = df[bird_cols].values
    for i in range(0,len(preds),12):
        preds_soundscape = preds[i:i+12]
        max_preds_soundscape = preds_soundscape.max(0, keepdims=True)
        max_preds_soundscape = max_preds_soundscape + (preds_soundscape.mean() - max_preds_soundscape.mean())
        preds_soundscape = preds_soundscape + 0.5 * max_preds_soundscape
        preds[i:i+12] = preds_soundscape
    df[bird_cols] = preds

    # Min-max normalization and power scaling
    for bird_col in bird_cols:
        df[bird_col] = df[bird_col].values - df[bird_col].min()
        df[bird_col] = df[bird_col].values / df[bird_col].max()
    df[bird_cols] = df[bird_cols].values ** p
    return df
The final ensemble achieved a public score of 0.924 and a private score of 0.922.

Project Details
Authors
RihanPiggy
honglihang

yokuyama
yokuyama


Share
0 Comments
Hotness
Skip to
content
Kaggle

Create
Search

Home

Competitions

Datasets

Models

Benchmarks

Game Arena

Code

Discussions

Learn

More


View Active Events

Skip to
content
Kaggle

Sign In

Register
9th place solution

BirdCLEF+ 2025

Solution Writeup · 9th place · Jun 7, 2025

Thanks to host and every participant in this competion! And thanks to my great teammate @yuanzhezhou ! I learn a lot from this interesting contest and I am glad to get my first gold medal.
Our solution are as follows.

1.Dataset
Only train_audio and train_soundscapes of 2025

2.Data preprocessing
We remove 50% human voice in the audio. Because we find that removing all the human voice will hurt model performance.

3.Training
Stage 1 model
Sample
In our experiments, rms sampling is better than random sampling.

Model
We use sed models opened by 2nd place solution of 2023. Thanks for @honglihang 's fabulous models, I just need to modify some configuration and code so that I got a single model with 0.850+ on lb.

Besides, we also use the cnn model in the public notebook as our first-stage model.

Loss
For sed models, we use FocalBCE loss in the training.
For cnn models, we use CE+BCE loss. In the early stage of training, we use ce loss which can improve convergence speed and use bce loss in the late stage of training.

Data Augmentation
For raw signal:

PitchShift and Shift
Sumix proposed by 7th place solution of 2023 with 0.5 probability
For Mel-Spectrogram:

Mixup2
Time masking
FilterAugment with 0.5 probability
FrequencyMasking with 0.5 probability
PinkNoise with 0.5 probability
Mel-Spectrogram parameters
target_duration  = 5
img_size = 384
SR = 32000
n_fft = 2048
n_mels = 256
f_min = 20
f_max = 16000
hop_length = target_duration * SR // (img_size - 1)

melspec_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=SR,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max,
            n_fft=n_fft,
            pad_mode="constant",
            norm="slaney",
            onesided=True,
            mel_scale="htk",
        )
db_transform = torchaudio.transforms.AmplitudeToDB(
            stype="power", top_db=80
        )

# for cnn
SR = 32000
n_fft = 2048
n_mels = 128
f_min = 20
f_max = 14000
hop_length = 1024
EMA
Applying ema in the training makes my model more robust and narrows the gap between different epochs.

Backbone
For sed models, we use efficientnetv2_b3, eca_nfet_l0 and seresnext26t and train in 10s segments. For cnn model, we use efficientnet_b0 and train in 5s segments.

By blending these models, we can get a sed model with 0.888 on lb and a cnn model with 0.840+ on lb.

Stage 2 model
We use our sed, cnn and models from public notebook to generate pseudo labels in every 10-second chunk. For every sample, we only use top 10% classes as soft labels and other classes are set as zeros.And then, we use the same pipeline to train new sed models. In this stage, we select efficientnetv2_b3 and seresnext26t as our final models' backbone.

By using pseudo labels, we got 0.02+ boost on lb.

4. Inference and TTA
We put 10s chunk into model and use 2 seconds as window length to apply TTA. You can check for more details in my inference notebook. All models were transformed to onnx format firstly.

5. Ensemble and Post-processing
We didn't have too much time to adjust ensemble and post-processing strategy, so we just use weighted average to ensemble models.

For post-processing:

smooth prediction with [0.2, 0.6, 0.2] (boost 0.005~0.01 on lb)
fmax - 2000 in the inference (boost 0.002 on lb)
6. Model score
We notice that random seed make a relatively huge difference on results. So we ensemble models with different seeds and same pipeline.

backbone	seed	public	private
efficientnetv2_b3 ①	42	0.903	0.913
efficientnetv2_b3 ②	3407	0.904	0.915
efficientnetv2_b3 ③	2025	0.896	0.915
seresnext26t ④	42	0.899	0.908
By ensemble ①、②、④, we got 0.913 on lb and 0.921 on private, and we select it as our final submission.
By ensemble ①、②、③, we got 0.913 (lower) on lb and 0.922 on private, but we miss this submission.

Actually, in our final chance submission, we tried to emsemble all 4 models and it just took about 1m20s on ten 60-second soudscapes. It seemed that it won't time out but it did.

BUT when we used single thread instead of multi-threads to submit it again after the contest is end, it worked well and got 0.924 on private. It seems that onnx+multi-threads will lead to bug, which makes us miss the chance to obtain top 5 :(

7. Some ideas don't work on lb but do work on private.
Because of the instability on lb, we regrettably miss some ideas that don't work on lb but do work well on private.

Resample pseudo labels by probility.
With window length of 10 seconds and step of 3 seconds, models predict every 60-second soudscape. Then, we get 17 segments on every soudscape and select the top 3 segments with the highest sum of probability.
This idea makes me get a model with 0.891 on lb but 0.916 on private.

Adjust probility by max and mean in post-processing
This idea comes from 3rd place solution of 2024.

def max_mean_post(sub_df, cfg):
    print("Postprocessing submission predictions...")
    sub_df.loc[:, cfg.bird_cols] = logit(sub_df.iloc[:, 1:].values)
    sub_df["group"] = sub_df['row_id'].str.rsplit('_', n=1).str[0]

    dfg_max = sub_df[["group"] + cfg.bird_cols].groupby('group').max().reset_index()
    dfg_mean = sub_df[["group"] + cfg.bird_cols].groupby('group').mean().reset_index()

    delta = dfg_mean[cfg.bird_cols].mean(1) - dfg_max[cfg.bird_cols].mean(1)
    for c in cfg.bird_cols:
        dfg_max[c] += delta

    sub_df = sub_df.merge(dfg_max, how="left", on="group", suffixes=('', '_delta'))
    for c in cfg.bird_cols:
        sub_df[c] = expit((sub_df[c] + sub_df[c + "_delta"]) / 2)

    return sub_df.loc[:, ['row_id']+cfg.bird_cols].reset_index(drop=True)
It decrease 0.007 on lb but boost 0.001~0.003 on private.

8. What did not work
raw signal model
ensemble with rank average
rms sample with energy weight
use common name as auxiliary target
lower rank power postprocessing
add guassian noise into raw signal
Thanks to everyone! There are many ideas that I haven't had time to try yet. Hope that I can validate them in the BirdCLEF 2026!

Inference notebook: https://www.kaggle.com/code/i2nfinit3y/bird2025-9th-place-solution?scriptVersionId=244060530

Project Details
Authors
yuanzhe zhou
yuanzhezhou

I2nfinit3y
i2nfinit3y


Share
12 Comments
Hotness
Greg L
Posted 5 months ago

Congratulations!! Really amazing work + thanks a lot for open sourcing the sed models.
Is it possible for you to also open source the training framework? I am new to kaggle competitions and would like to learn from your work, especially by comparing it to the open source 2023 2nd place solution and see what have changed. I tried to train models from scratch using the 2023 repo but somehow it does not work out well for this year's data.


Reply
I2nfinit3y
Topic Author
Posted 5 months ago

9th in this Competition

I have not yet arranged my code. But there are some points that boost my model performance a lot:

CosineAnnealingLR1
PitchShift and Shift for raw signal
Sumix for raw signal
Mixup2 for Mel-Spectrogram
FilterAugment for Mel-Spectrogram
FocalBCE

Reply
Greg L
Posted 5 months ago

Thank you for your responses! Really helpful.


Reply
MYSO
Posted 6 months ago

5th in this Competition

Thank you for sharing your solution — it’s very helpful and informative.
Regarding the line “We remove 50% human voice in the audio,” do you mean that you removed half of the files that contained human voice, or that you removed half of the human voice segments within each file?


Reply
I2nfinit3y
Topic Author
Posted 6 months ago

9th in this Competition

The half of the human voice segments within each file


Reply
MYSO
Posted 6 months ago

5th in this Competition

Got it, thanks for the clarification!


Reply
thacrobatheskis
Posted 6 months ago

710th in this Competition

Congrats, and thanks for sharing!

Given the high variance in submissions, how did you test different ideas? Did you make just a single submission to test an idea, or did you aggregate the LB score over multiple seeds? Or did you have a way to local validate ideas before submitting them?


Reply
I2nfinit3y
Topic Author
Posted 6 months ago

9th in this Competition

Just submit 2 models with different seeds and test ideas by lb


Reply
Jan H.
Posted 6 months ago

Well done! How did you select train_audio segments? All segments, random, first and last or something else? Also, did you change the process for that over time?


Reply
I2nfinit3y
Topic Author
Posted 6 months ago

9th in this Competition

I select segments by rms, just like this discussion


Reply
Overvalue Awareness
Posted 6 months ago

Congrats!
How many hours did you take to train models on stage 1 and 2 respectively? Did you use only GPU from Kaggle?


Reply
I2nfinit3y
Topic Author
Posted 6 months ago

9th in this Competition

3090 with about 2 hours. If with gpu of kaggle, it takes about 4 hours.


Reply
CodeHacker
Posted 6 months ago

591st in this Competition

First of all, congratulations! How did you decide which 50% of the human voice to remove? Was it random, or based on some criteria?


Reply
I2nfinit3y
Topic Author
Posted 6 months ago

9th in this Competition

Just randomly


Reply
CodeHacker
Posted 6 months ago

591st in this Competition

Got it, thanks! Then may I ask how you detected or identified the human voice segments in the first place?


Reply
I2nfinit3y
Topic Author
Posted 6 months ago

9th in this Competition

I got human voice sements from this public notebook


Reply
CodeHacker
Posted 6 months ago

591st in this Competition

Thank you!


Reply
Skip to
content
Kaggle

Create
Search

Home

Competitions

Datasets

Models

Benchmarks

Game Arena

Code

Discussions

Learn

More


View Active Events

Skip to
content
Kaggle

Sign In

Register
10th Solution

BirdCLEF+ 2025

Solution Writeup · 10th place · Jun 6, 2025

Thanks to the organizers for hosting this competition, and also to the many Kaggler participants from previous years whose solutions have provided me with great inspiration.
BirdCLEF is a challenging competition, Although the dataset shift isn't extremely large, the lack of a validation set from the same distribution makes training and evaluation quite difficult.
I adopted the following approach during the competition:

1. Establishing Baseline Model Performance
I used the entire train_audio set and validated directly on the leaderboard. I found the following model architectures performed best:

(SED + CE loss) > (SED + FocalBCE) > (CNN + FocalBCE) > (CNN + CE)
This was just an initial experiment, so it may have been influenced by other factors and not be generally applicable. But due to limited online submission opportunities, I stuck with SED + CE loss for the rest of the experiments.
Models trained in this stage are referred to as stage1 models.

2. Domain Adaptation & pseudo-labeled data
As seen in previous competitions, leveraging unlabeled data (train_soundscapes) can lead to significant gains.
I used the following method:

Use the stage1 models to predict on train_soundscapes, generating soft labels
Apply high and low thresholds (thresh_low, thresh_high) to extract confident negative and positive samples
These samples (assumed to be relatively clean) were added to train_audio to train new models — the stage2_models. The ambiguous samples in between were excluded
Predict again on train_soundscapes using stage2_models to get updated soft labels
Through this iterative process, we obtained relatively clean pseudo-labeled data, referred to as pp_data_clean.
3. Data Processing
Following community discussions, I removed samples with obvious human speech — this cleaned dataset is called train_audio_clean.
To further improve diversity and make data clean, I used trained models to remove segments that clearly did not contain the target species, sampling at 5-second intervals, forming a new dataset: train_audio_clean_v2.
Both datasets were used for training.

4. Mel Spectrogram Parameters
I used mel spectrograms with the following resolutions:

384x160
384x256
320x192
320x160
fmin = 0, fmax = 16000， n_fft=1536/2048
5. Loss Function
I adopted a modified hybrid loss with CE.
Although public notebooks showed that ConvNeXt performed well, using standard CE loss led to a lot of noisy predictions (many false positives).
So, I gradually incorporated the following loss to better stabilize training:

threshold = -5
neg_mask = (y == 0.0)
negative_logits = logits[neg_mask]
penalty = F.relu(negative_logits - threshold)
sorted_penalty, _ = torch.sort(penalty)
cutoff_index = int(len(sorted_penalty) * 0.95)
selected_penalty = sorted_penalty[:cutoff_index]
mean_penalty = selected_penalty.mean()
loss = ce_loss + 0.1 * mean_penalty
In some models, I also penalized positive samples with confidence scores that were too low.

6. Post Processing
smooth with kernel: [0.02, 0.08, 0.8, 0.08, 0.02]
and average:
def get_mean_scales(ref_freq):
  alpha_max = 0.3
  alpha_min = 0.1
  # rare classed more recall
  alpha = alpha_min + ref_freq * (alpha_max - alpha_min)
  return alpha
...
 for c in len(n_classes):
   a = alpha[c]
   pred_prob[:, c] = pred_prob[:, c] * (1-a) + pred_prob[:, c].mean(keepdims=True) * a
7. Some Model Results
ConvNeXt Tiny (320x192): Public: 0.901 Private: 0.914
3x ConvNeXt Tiny Ensemble: Public: 0.908 Private: 0.921
EfficientNetV2-S (384x160) Public: 0.904 Private: 0.906
EfficientNetV2-S (384x256) Public: 0.896. Private: 0.914
Diversity was controlled by varying the dataset used and the proportion and quality of mixed-in pp_clean_data.
I ultimately ensembled 10 models in total, Public: 0.915 Private: 0.921 .
However, due to the instability of the leaderboard (LB), I wasn’t able to select the optimal combination of models.
8. Other Details
10s/15s for train, infer on 5s
AdamW + CosineAnnealingLR， LR： 2/3*e-4, WD: 1e-3/1e-4
Mixup with additive, frequency, and time masking
With and without noise augmentation
Many other common techniques like label smoothing, flip, etc.
Project Details
Author
lhwcv
lihaoweicvch


Share
6 Comments
Hotness
Overvalue Awareness
Posted 6 months ago

Congrats!
How many hours do you spend for training those 10 models in total? (and for individual?)
And did you use only GPU from Kaggle? If from the outside how much did it cost approximately?
Thank you.


Reply
lhwcv
Topic Author
Posted 6 months ago

10th in this Competition

1 model cost about 2hour for me with RTX 5090


Reply
Salman Ahmed
Posted 6 months ago

26th in this Competition

Congrats on the solo gold!


Reply
1
Athar Sayed
Posted 6 months ago

564th in this Competition

Thanks for sharing and congrats on solo gold !


Reply
1
coolz
Posted 6 months ago

19th in this Competition

Congrats on the solo gold!


Reply
1
Ty-Yuki
Posted 6 months ago

754th in this Competition

Congrats on the solo gold!


Reply
1
Skip to
content
Kaggle

Create
Search

Home

Competitions

Datasets

Models

Benchmarks

Game Arena

Code

Discussions

Learn

More


View Active Events

Skip to
content
Kaggle

Sign In

Register
11th solution

BirdCLEF+ 2025

Solution Writeup · 11th place

Thank you to the organizers for organizing such an interesting competition, and also thank the excellent plans from previous sessions for inspiring me. Also, congratulations to all the top winners. Next, I will introduce my solution.

1、Training Data
The 206 categories have been expanded to 316
(1) 25 year competition data: train_audio and train_soundscapes

(2) select 110 categories (categories with a sample size of less than 10) from the data of previous competitions

This additional category is mainly for building local cv. Unfortunately, this cv strategy is ineffective. However, this mixed training improved my lb, so I maintained this operation

The maximum sample size for each category is 500. Categories smaller than 10 will be upsampled
Only remove human voices from part of the CAS data
2、Model Architecture
Sed models opened by 2nd place solution of 2023. I added the code related to pseudo-labels on this basis

backbones are：

tf_efficientnetv2_b3
tf_efficientnetv2_s
All of them are trained on 10sec clip.

3、Loss function
Using ce loss, compared with BCE loss, there is a qualitative improvement (0.83→0.88) in multiple models.

4、Mel Spectrogram Parameters
{'sample_rate': 32000, 'n_mels': 256, 'image_size': 300, 'f_min': 90, 'f_max': 14000, 'n_fft': 1536, 'normalized': True, 'hop_length': 535}

{'sample_rate': 32000, 'n_mels': 256, 'image_size': 300, 'f_min': 50, 'f_max': 14000, 'n_fft': 1024, 'normalized': True, 'hop_length': 535}
5、pseudo label
Select high-quality pseudo-labels based on the entropy-based screening strategy mentioned in rank 10 of last year 24
# 示例代码
import numpy as np
epsilon = 1e-12

pre_probs = sub_df[columns].values
print(pre_probs.shape)

probs = pre_probs.copy()
entropies = -np.sum(probs * np.log(probs + epsilon), axis=1)
print(entropies.shape)

# 筛选前 20% 的伪标签
top_10_indices = np.argsort(entropies)[:int(len(entropies) * 0.2)]
top_10_pseudo_probs = probs[top_10_indices]
print(top_10_pseudo_probs.shape)

# 对于每个类别的伪标签，将低于前 92% 的标签值设置为 0
for i in range(top_10_pseudo_probs.shape[1]):
    class_probs = top_10_pseudo_probs[:, i]
    threshold = np.percentile(class_probs, 92)
    top_10_pseudo_probs[class_probs < threshold, i] = 0
print(top_10_pseudo_probs.shape)
The real dataset and the pseudo-label dataset are concatenated into the model, and the loss of the pl part is down weighted
batch_size = 96，pl_batch_size = 16

6、Ensemble and Post-processing
ensemble model
5 ✖️ v2b3 + 1✖️v2s

Public Score: 0.920
Private Score: 0.919

post precessing
The post-processing is the same as the rank 6 in 2024，An increase of approximately 0.001

def smooth_array_general(array, w=[0.1, 0.2, 0.4, 0.2, 0.1]):
    smoothed_array = np.zeros_like(array)
    timesteps = array.shape[0]
    radius = len(w) // 2

    for t in range(timesteps):
        for i, weight in enumerate(w):
            index = t - radius + i
            if index < 0:
                smoothed_array[t] += array[0] * weight
            elif index >= timesteps:
                smoothed_array[t] += array[-1] * weight
            else:
                smoothed_array[t] += array[index] * weight
    for c in range(array.shape[1]):
        smoothed_array[:, c] = smoothed_array[:, c] * 0.8 + smoothed_array[:, c].mean(keepdims=True) * 0.2
    return smoothed_array
7、not work
CNN
rms sample
remove all human voices
Project Details
Author
Baiph
pingfan


Share
3 Comments
Hotness
shanzhong8
Posted 6 months ago

3rd in this Competition

Congrats on your solo win, bro!


Reply
1
Jan H.
Posted 6 months ago

Congrats! If rms sampling didn’t work, how did you sample the training data?


Reply
Baiph
Topic Author
Posted 6 months ago

11th in this Competition

random sample


Reply

