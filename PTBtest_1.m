%% PTB Initialize
close all;
sca;
clear;
PsychDefaultSetup(2);

%% setting your parameters for your experiment
sessions = 3;
trials = 6;
frequencies = [9, 11, 13];



%parpool('local', 4);
rng('default');
rng('shuffle');
%channels = [1,2,3,4:5,6]';
channels = 1:10;%max 26 channels per time
buffersize = 2000;
buffer = zeros(length(channels), buffersize);
harmonics = 3;
REST=1;
EXETIME = 5;  %unit:s
%%%%%%% lsl
% create a new outlet
lib = lsl_loadlib();
% resolve a stream...
disp('Resolving an EEG stream...');
result = {};

%create a new outlet for labels
info = lsl_streaminfo(lib,'SSVEPMarkerStream','Markers',1,0,'cf_string','myuniquesourceid23443');
outlet = lsl_outlet(info);

%  create a new inlet for results
% resolve a stream...
disp('Resolving a Markers stream...');
result = {};
while isempty(result)
    result = lsl_resolve_byprop(lib,'name','ResultMarker'); 
end
% create a new inlet
disp('Opening an inlet...');
inlet = lsl_inlet(result{1});
disp('Now receiving data...');

% define the screen size and square shape 
screenNumber = max(Screen('Screens'));

white = WhiteIndex(screenNumber);
grey = white / 2;
black = BlackIndex(screenNumber);

spaceKey = KbName('space');
leftKey = KbName('leftarrow');
rightKey = KbName('rightarrow');
escKey = KbName('escape');

baseRect = [0 0 300 300];
waitFrames = 1;

[win, winRect] = PsychImaging('OpenWindow', screenNumber, [0 0 0 0], [0 0 1920 1080]);%[0 0 1920 1080]
[centerXwin, centerYwin] = RectCenterd(winRect);
oldFontSize=Screen(win,'TextSize', min(round(winRect(4)*1/10),64));
 

%% start to code
texts = ['ABCDEFGHIJKLMN'];        
fai = 0;
luminance = @(f, fai, vbl) 0.5 * (1 + sin(2 * pi * f * vbl + fai));

% generate rects
nfreqs = size(frequencies, 2);
rects = zeros(4, nfreqs);
for i = 1:nfreqs
    rects(:, i) = CenterRectOnPointd(baseRect, 2*i*centerXwin/(nfreqs+1), centerYwin);
end

vbl = Screen('Flip', win);
ifi = Screen('GetFlipInterval', win);

for session = 1:1:sessions
    % waiting the keyboard interupt
    DrawFormattedText(win, 'Press any key to start a session!', 'center', 'center', white);
    vbl=Screen('Flip', win);
    KbStrokeWait;

    % begin sessions
    turn = randperm(trials);
    for trial=1:trials	
        % randomly choose the target 
        Screen('FillOval', win, [1, 1, 1], rects(:,mod(turn(trial),nfreqs)+1));
        vbl=Screen('Flip', win);
        if KbCheck
            sca;
            break;
        end

        pause(2);
        Screen('Flip', win);
        pause(0.5);
        outlet.push_sample({'trial start'});
        
        % you can end up this session with KbCheck or until frames exceed the EXEFRAME
        %
        frames = 0;
        tic;
        last = toc;
        while last<EXETIME
            % start paint tasks
            colors = ones(3, nfreqs);                
            for i = 1:nfreqs
                fai = pi/4*i;
                colors(:, i) = colors(:, i) * luminance(frequencies(i), fai, vbl);
            end

            Screen('FillRect', win, colors, rects);
            Screen('DrawingFinished', win);

            vbl = Screen('Flip', win, vbl + (waitFrames - 0.5) * ifi);
            ifi = Screen('GetFlipInterval', win);
            frames = frames + 1;
            last = toc;
        end
        A=1
        last = toc
        vbl=Screen('Flip', win);
        [mrks,ts] = inlet.pull_sample()
        
        DrawFormattedText(win, mrks{1}, 'center', 'center', white);
        vbl=Screen('Flip', win);
        pause(0.5);
        %black the screen and rest for a while 
        DrawFormattedText(win, '+', 'center', 'center', white);
        vbl=Screen('Flip', win);
        pause(1.5);
    end
end
outlet.push_sample({'end'});
sca;