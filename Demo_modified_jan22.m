%  Matlab Code-Library for Feature Selection
%  A collection of S-o-A feature selection methods
%  Version 6.2 October 2018
%  Support: Giorgio Roffo
%  E-mail: giorgio.roffo@glasgow.ac.uk
%
%  Before using the Code-Library, please read the Release Agreement carefully.
%
%  Release Agreement:
%
%  - All technical papers, documents and reports which use the Code-Library will acknowledge the use of the library as follows:
%    The research in this paper use the Feature Selection Code Library (FSLib) and a citation to:
%  ------------------------------------------------------------------------
% @InProceedings{RoffoICCV17,
% author={Giorgio Roffo and Simone Melzi and Umberto Castellani and Alessandro Vinciarelli},
% booktitle={2017 IEEE International Conference on Computer Vision (ICCV)},
% title={Infinite Latent Feature Selection: A Probabilistic Latent Graph-Based Ranking Approach},
% year={2017},
% month={Oct}}
%  ------------------------------------------------------------------------
% @InProceedings{RoffoICCV15,
% author={G. Roffo and S. Melzi and M. Cristani},
% booktitle={2015 IEEE International Conference on Computer Vision (ICCV)},
% title={Infinite Feature Selection},
% year={2015},
% pages={4202-4210},
% doi={10.1109/ICCV.2015.478},
% month={Dec}}
%  ------------------------------------------------------------------------

% FEATURE SELECTION TOOLBOX v 6.2 2018 - For Matlab 
% Please, select a feature selection method from the list:
% [1] ILFS 
% [2] InfFS 
% [3] ECFS 
% [4] mrmr 
% [5] relieff 
% [6] mutinffs 
% [7] fsv 
% [8] laplacian 
% [9] mcfs 
% [10] rfe 
% [11] L0 
% [12] fisher 
% [13] UDFS 
% [14] llcfs 
% [15] cfs 
% [16] fsasl 
% [17] dgufs 
% [18] ufsol 
% [19] lasso 

% Before using the toolbox compile the solution:
% make;

%% DEMO FILE
fprintf('\nFEATURE SELECTION TOOLBOX v 6.2 2018 - For Matlab \n');

%% Include dependencies
addpath('./lib'); % dependencies
addpath('./methods'); % FS methods
addpath(genpath('./lib/drtoolbox'));

%% Select a feature selection method from the list
listFS = {'ILFS','InfFS','ECFS','mrmr','relieff','mutinffs','fsv','laplacian','mcfs','rfe','L0','fisher','UDFS','llcfs','cfs','fsasl','dgufs','ufsol','lasso'};

% Method 1
[ methodID ] = readInput( listFS );

% Method 2
% methodID = usr_input;

selection_method = listFS{methodID}; 

%% Load the data and select features for classification
series = 2;
path = '/Users/binhnguyen/Library/Mobile Documents/com~apple~CloudDocs/Desktop/Digital Mental Health/3. Data and Analysis';
removed_var = {'MH_15A','MH_15B','MH_15C','MH_15D','MH_15E','MH_15F','MH_15G','ANXDVSEV','ANXDVGAD','ANXDVGAC'};

switch series
    case 1
        filename = '/Series_1/cpss-5311-E-series1_F1.csv';
        tbl = readtable (strcat(path,filename));
        Y = tbl.BH_30; % Perceived Mental Health
        X = tbl;
        X.BH_30 = [];
        
    case 2
        filename = '/Series_2/cpss-5311-E-series2_F1.csv';
        tbl = readtable (strcat(path,filename));
        Y = tbl.ANXDVSEV; % GAD7
        X = tbl (:,2:end-2);
        for i=1:length(removed_var)
            var = find (string(X.Properties.VariableNames) == string(removed_var(i)));
            X(:,var) = [];
        end
        
    case 4
        filename = '/Series_4/cpss-5311-E-sources_F1.csv';
        tbl = readtable (strcat(path,filename));
        Y = tbl.ANXDVSEV; % GAD7
        X = tbl (:,2:end-2);
        for i=1:length(removed_var)
            var = find (string(X.Properties.VariableNames) == string(removed_var(i)));
            X(:,var) = [];
        end
end

X_final = table2array (X);
Y_final = Y;

% Removing the 99 state
X_final (find (Y_final == 9),:) = [];
Y_final (find (Y_final == 9)) = [];

% No symptons and minimal should be the same
Y_final (find (Y_final == 0)) = 1;




%%%% TESTING BEG

% Combining 0,1,2 into one label
% Y_final (find (Y_final == 2)) = 1;
% Combining 3,4 into one label
% Y_final (find (Y_final == 3)) = 4;

% No symptons and minimal should be the same
Y_final (find (Y_final == 0)) = 1;
Y_final (find (Y_final == 0)) = 1;

% Focus on None vs Severe
xlab = cell(1,1);
ylab = cell(1,1);
for i =1:4
    xlab{i} = (X_final (find (Y_final == i),:));    
    ylab{i} = (Y_final (find (Y_final == i)));
end

% Final dataframes
X_final = [xlab{1};xlab{2};xlab{3};xlab{4}];
Y_final = [ylab{1};ylab{2};ylab{3};ylab{4}];

%%%% TESTING ENDING



df_final = [X_final Y_final];

P = cvpartition(Y_final,'Holdout',0.20);

X_train = double(X_final(P.training,:));
Y_train = (double( Y_final(P.training))); 
X_test = double( X_final(P.test,:) );
Y_test = (double( Y_final(P.test) ));


% Variable header names
feat_names = X.Properties.VariableNames;

% number of features
numF = size(X_train,2);

%% Feature Selection on training data
switch lower(selection_method)
    case 'inffs'
        % Infinite Feature Selection 2015 updated 2016
        alpha = 0.5;    % default, it should be cross-validated.
        sup = 1;        % Supervised or Not
        [ranking, w] = infFS( X_train , Y_train, alpha , sup , 0 );
        
    case 'ilfs'
        % Infinite Latent Feature Selection - ICCV 2017
        [ranking, weights] = ILFS(X_train, Y_train , 6, 0 );
        
    case 'fsasl'
        options.lambda1 = 1;
        options.LassoType = 'SLEP';
        options.SLEPrFlag = 1;
        options.SLEPreg = 0.01;
        options.LARSk = 5;
        options.LARSratio = 2;
        nClass=2;
        [W, S, A, objHistory] = FSASL(X_train', nClass, options);
        [v,ranking]=sort(abs(W(:,1))+abs(W(:,2)),'descend');
    case 'lasso'
        lambda = 25;
        B = lasso(X_train,Y_train);
        [v,ranking]=sort(B(:,lambda),'descend');
        
    case 'ufsol'
        para.p0 = 'sample';
        para.p1 = 1e6;
        para.p2 = 1e2;
        nClass = 2;
        [~,~,ranking,~] = UFSwithOL(X_train',nClass,para) ;
        
    case 'dgufs'
        
        S = dist(X_train');
        S = -S./max(max(S)); % it's a similarity
        nClass = 2;
        alpha = 0.5;
        beta = 0.9;
        nSel = 2;
        [Y,L,V,Label] = DGUFS(X_train',nClass,S,alpha,beta,nSel);
        [v,ranking]=sort(Y(:,1)+Y(:,2),'descend');
        
        
    case 'mrmr'
        ranking = mRMR(X_train, Y_train, numF);
        
    case 'relieff'
        [ranking, w] = reliefF( X_train, Y_train, 20);
        
    case 'mutinffs'
        [ ranking , w] = mutInfFS( X_train, Y_train, numF );
        
    case 'fsv'
        [ ranking , w] = fsvFS( X_train, Y_train, numF );
        
    case 'laplacian'
        W = dist(X_train');
        W = -W./max(max(W)); % it's a similarity
        [lscores] = LaplacianScore(X_train, W);
        [junk, ranking] = sort(-lscores);
        
    case 'mcfs'
        % MCFS: Unsupervised Feature Selection for Multi-Cluster Data
        options = [];
        options.k = 5; %For unsupervised feature selection, you should tune
        %this parameter k, the default k is 5.
        options.nUseEigenfunction = 4;  %You should tune this parameter.
        [FeaIndex,~] = MCFS_p(X_train,numF,options);
        ranking = FeaIndex{1};
        
    case 'rfe'
        ranking = spider_wrapper(X_train,Y_train,numF,lower(selection_method));
        
    case 'l0'
        ranking = spider_wrapper(X_train,Y_train,numF,lower(selection_method));
        
    case 'fisher'
        ranking = spider_wrapper(X_train,Y_train,numF,lower(selection_method));
        
        
    case 'ecfs'
        % Features Selection via Eigenvector Centrality 2016
        alpha = 0.5; % default, it should be cross-validated.
        ranking = ECFS( X_train, Y_train, alpha )  ;
        
    case 'udfs'
        % Regularized Discriminative Feature Selection for Unsupervised Learning
        nClass = 2;
        ranking = UDFS(X_train , nClass );
        
    case 'cfs'
        % BASELINE - Sort features according to pairwise correlations
        ranking = cfs(X_train);
        
    case 'llcfs'
        % Feature Selection and Kernel Learning for Local Learning-Based Clustering
        ranking = llcfs( X_train );
        
    otherwise
        disp('Unknown method.')
end

%% Number of features to take
k = numF;
% k=20;

%% SVM classifier 

Mdl_SVM = fitcecoc(X_train(:,ranking(1:k)),Y_train);
C_SVM = predict(Mdl_SVM,X_test(:,ranking(1:k)));
err_SVM = sum(Y_test~= C_SVM)/P.TestSize; % mis-classification rate
conMat_SVM = confusionmat(Y_test,C_SVM); % the confusion matrix

% ranking
% disp (string(feat_names (ranking (1:k))));

fprintf('\nMethod %s (Linear-SVMs): Accuracy: %.2f%%, Error-Rate: %.2f \n',selection_method,100*(1-err_SVM),100*(err_SVM));




%% DT classifier 

Mdl_DT = fitctree(X_train(:,ranking(1:k)),Y_train);
C_DT = predict(Mdl_DT,X_test(:,ranking(1:k)));
err_DT = sum(Y_test~= C_DT)/P.TestSize; % mis-classification rate
conMat_DT = confusionmat(Y_test,C_DT); % the confusion matrix

% ranking
% disp (string(feat_names (ranking (1:k))));

fprintf('\nMethod %s (DT): Accuracy: %.2f%%, Error-Rate: %.2f \n',selection_method,100*(1-err_DT),100*(err_DT));

% MathWorks Licence
% Copyright (c) 2016-2017, Giorgio Roffo
% All rights reserved.
%
% Redistribution and use in source and binary forms, with or without
% modification, are permitted provided that the following conditions are
% met:
%
%     * Redistributions of source code must retain the above copyright
%       notice, this list of conditions and the following disclaimer.
%     * Redistributions in binary form must reproduce the above copyright
%       notice, this list of conditions and the following disclaimer in
%       the documentation and/or other materials provided with the distribution
%     * Neither the name of the University of Verona nor the names
%       of its contributors may be used to endorse or promote products derived
%       from this software without specific prior written permission.
%
% THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
% AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
% IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
% ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
% LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
% CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
% SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
% INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
% CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
% ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
% POSSIBILITY OF SUCH DAMAGE.
