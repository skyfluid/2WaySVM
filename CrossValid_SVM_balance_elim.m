clc
clear
close all

%cell_sel = [1, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 23, 24, 30, 33, 39, 40, 41, 42, 44, 45, 46, 47, 49, 50, 51, 53, 57, 60, 65, 66, 67, 68, 69, 70, 73, 74, 75, 76, 77, 78, 80, 82, 83, 84, 86, 87, 88, 89, 93, 94, 95, 97, 102, 103, 104, 105, 111, 112, 113, 114, 118, 120, 121, 129, 130, 131, 132, 134, 135, 136, 138, 141, 144, 145, 151, 154, 155, 157, 158, 159, 166, 167, 168, 169, 170, 171, 174, 181, 189, 193];
%cell_sel = [5, 6, 7, 8, 9, 23, 42, 43, 55, 57, 60, 61, 62, 66, 67, 71];
cell_sel = [1:199];

def_bin_size = 50;
def_cell_count = length(cell_sel);
def_max_pc_count = 400;
def_plot_pc_count = 50;
def_balance_coef = 1;

load(['../bindatagen/binMatWord_' int2str(def_bin_size) '.mat'])

wordtoVerify = {'CLAW' 'CRICKET' 'FLAG' 'FORK' 'LION' 'MEDAL' 'OYSTER' 'SERPENT' 'SHELF' 'SHIRT'};
mkdir('PCA_SVM_balance/')

trialSet = [];
wordSet = [];

for iTrial=1:length(binMat)
    A = binMat{iTrial}.mat(cell_sel,:)';
    B = binMat{iTrial}.type;
    trialSet = [trialSet; A(:)'];
    wordSet = [wordSet; B];
end

outfile = fopen('datalog.txt','w');


stat_total = {};
for nPC=1:def_max_pc_count
for iWord=1:length(wordtoVerify)
    stat_total{iWord}.word = wordtoVerify{iWord};
    
    SVM_candidate = [];
    
    for iTestSet=1:10
        testBegin = (10-iTestSet)*80+1;
        testEnd = (11-iTestSet)*80;
        testSet = trialSet(testBegin:testEnd,:);
        testSetw = wordSet(testBegin:testEnd,:);
        trainingSet = [trialSet(1:testBegin-1,:); trialSet(testEnd+1:end,:)];
        trainingSetw = [wordSet(1:testBegin-1,:); wordSet(testEnd+1:end,:)];

        local_best_score = 0;
        local_best_scorep = 0;
        for iWorkSet=1:9
            verifyBegin = (9-iWorkSet)*80+1;
            verifyEnd = (10-iWorkSet)*80;
            verifySet = trainingSet(verifyBegin:verifyEnd,:);
            verifySetw = trainingSetw(verifyBegin:verifyEnd,:);
            workSet = [trainingSet(1:verifyBegin-1,:); trainingSet(verifyEnd+1:end,:)];
            workSetw = [trainingSetw(1:verifyBegin-1,:); trainingSetw(verifyEnd+1:end,:)];
            
            % balance processing
            posRowSel = find(strcmp(wordtoVerify{iWord}, workSetw) == 1)';
            posRows = workSet(posRowSel, :);
            posRowsw = workSetw(posRowSel, :);
            
            workSet(posRowSel, :) = [];    % remove the rows of word to be verified
            workSetw(posRowSel, :) = [];
            
            workSet = workSet(1:(length(posRowSel) * def_balance_coef), :);
            workSetw = workSetw(1:(length(posRowSel) * def_balance_coef), :);
            
            workSet = [workSet; posRows];
            workSetw= [workSetw; posRowsw];

            [coeff, score, latent] = pca(workSet);  % PCA

            varCov = cumsum(latent)./sum(latent);    % find the variance coverage for PCs
            varCov = varCov(nPC);
            %nPC = find(varCov > VarCovThreshold, 1);
            scoreSel = (workSet) * coeff(:, 1:nPC);
            distingSet = strcmp(wordtoVerify{iWord}, workSetw);

            options.MaxIter = 150000;
            svmStruct = svmtrain(scoreSel, distingSet, 'options', options, 'AutoScale', 'off'); %, 'showplot',true);
            %svmStruct = svmtrain(scoreSel, distingSet, 'options', options); %, 'showplot',true);
            resultSet = svmclassify(svmStruct, verifySet*coeff(:, 1:nPC)); %, 'showplot',true);
            ansSet = strcmp(wordtoVerify{iWord}, verifySetw);
            Accuracy = sum(eq(resultSet, ansSet))/length(ansSet);
            HitRate = sum(resultSet(find(ansSet)))/length(find(ansSet));

            if (Accuracy > local_best_score | ((Accuracy == local_best_score) & (HitRate > local_best_scorep)))
                local_best_score = Accuracy;
                local_best_scorep = HitRate;
                SVM_candidate{iTestSet}.svm = svmStruct;
                SVM_candidate{iTestSet}.pc = coeff(:, 1:nPC);
                SVM_candidate{iTestSet}.varcov = varCov;
            end
        end
        resultSet = svmclassify(SVM_candidate{iTestSet}.svm, testSet*SVM_candidate{iTestSet}.pc);
        ansSet = strcmp(wordtoVerify{iWord}, testSetw);
        Accuracy = sum(eq(resultSet, ansSet))/length(ansSet);
        HitRate = sum(resultSet(find(ansSet)))/length(find(ansSet));
        FHitRate = (length(find(ansSet==0)) - sum(resultSet(find(ansSet==0))))/length(find(ansSet==0));
        d_prime = min([norminv(0.99) norminv(HitRate)]) - max([norminv(0.01) norminv(1 - FHitRate)]);
        
        fprintf(outfile, 'Word: %s, SVM-%2d, #TestVec: %d, #HR: %f, #pHR: %f, #nHR: %f, #PC:%d\n', wordtoVerify{iWord}, iTestSet, length(ansSet), Accuracy, HitRate, FHitRate, nPC);
        stat_total{iWord}.data(nPC).HR(iTestSet) = Accuracy;
        stat_total{iWord}.data(nPC).pHR(iTestSet) = HitRate;
        stat_total{iWord}.data(nPC).nHR(iTestSet) = FHitRate;
        stat_total{iWord}.data(nPC).DP(iTestSet) = d_prime;       
        stat_total{iWord}.data(nPC).varcov(iTestSet) = SVM_candidate{iTestSet}.varcov;
        stat_total{iWord}.data(nPC).pc{iTestSet} = SVM_candidate{iTestSet}.pc;
        stat_total{iWord}.data(nPC).svm{iTestSet} = SVM_candidate{iTestSet}.svm;
    end
    stat_total{iWord}.avgHR(nPC) = mean(stat_total{iWord}.data(nPC).HR, 2);
    stat_total{iWord}.avgpHR(nPC) = mean(stat_total{iWord}.data(nPC).pHR, 2);
    stat_total{iWord}.avgnHR(nPC) = mean(stat_total{iWord}.data(nPC).nHR, 2);
    stat_total{iWord}.avgDP(nPC) = mean(stat_total{iWord}.data(nPC).DP, 2);
    stat_total{iWord}.avgvarcov(nPC) = mean(stat_total{iWord}.data(nPC).varcov, 2);
end

save(['stat_total_balance_' int2str(def_cell_count) '_' int2str(def_bin_size)  'ms.mat'], 'stat_total', '-v7.3')

end

fclose(outfile);

set(0, 'DefaultFigureVisible', 'on')

%load(['stat_total_balance_' int2str(def_cell_count) '_' int2str(def_bin_size)  'ms.mat']);

for iWord=1:length(wordtoVerify)
    avgHR = [];
    avgpHR = [];
    avgnHR = [];
    avgvarcov = [];
    dp = [];
    
    for nPC=1:def_plot_pc_count

        avgHR = [avgHR stat_total{iWord}.data(nPC).HR'];
        avgpHR = [avgpHR stat_total{iWord}.data(nPC).pHR'];
        avgnHR = [avgnHR stat_total{iWord}.data(nPC).nHR'];
        avgvarcov = [avgvarcov stat_total{iWord}.data(nPC).varcov'];
        
        dpp=[];
        for iTest=1:10            
            HR = min([norminv(0.99) norminv(stat_total{iWord}.data(nPC).pHR(iTest))]);
            FL = max([norminv(0.01) norminv(1 - stat_total{iWord}.data(nPC).nHR(iTest))]);
            dpp(iTest) = HR - FL;
        end
        dp = [dp dpp'];  % some elements may not contain trial


    end


    figure;
    boxplot(avgHR);
    title(['Accuracy of Two-Way SVM (balance) for "' wordtoVerify{iWord} '" (' int2str(def_cell_count) 'u ' int2str(def_plot_pc_count) 'pc)']);
    xlabel('Number of PCs'); ylabel('Accuracy');
    saveas(gcf, ['./PCA_SVM_balance/Accu_balance_' wordtoVerify{iWord} 'box.jpg'], 'jpg');

    figure;
    boxplot(avgpHR);
    title(['Accuracy (true) of Two-Way SVM (balance) for "' wordtoVerify{iWord} '" (' int2str(def_cell_count) 'u ' int2str(def_plot_pc_count) 'pc)']);
    xlabel('Number of PCs'); ylabel('Accuracy: hit/ (hit+miss)');
    saveas(gcf, ['./PCA_SVM_balance/AccuP_balance_' wordtoVerify{iWord} '_box.jpg'], 'jpg');

    figure;
    boxplot(avgnHR);
    title(['Accuracy (false) of Two-Way SVM (balance) for "' wordtoVerify{iWord} '" (' int2str(def_cell_count) 'u ' int2str(def_plot_pc_count) 'pc)']);
    xlabel('Number of PCs'); ylabel('Accuracy: CR/(CR+FA)');
    saveas(gcf, ['./PCA_SVM_balance/AccuN_balance_' wordtoVerify{iWord} '_box.jpg'], 'jpg');

    figure;
    boxplot(avgvarcov);
    title(['Variance Coverage of Two-Way SVM (balance) for "' wordtoVerify{iWord} '" (' int2str(def_cell_count) 'u ' int2str(def_plot_pc_count) 'pc)']);
    xlabel('Number of PCs'); ylabel('Variance Coverage');
    saveas(gcf, ['./PCA_SVM_balance/Varcov_balance_' wordtoVerify{iWord} '_box.jpg'], 'jpg');    

    figure;
    boxplot(dp);
    title(['d-prime for word ' wordtoVerify{iWord} ' (' int2str(def_cell_count) 'u ' int2str(def_plot_pc_count) 'pc) (balance)']);
    xlabel('Number of PCs'); ylabel('Variance Coverage');
    saveas(gcf, ['./PCA_SVM_balance/DP_balance_' wordtoVerify{iWord} '_box.jpg'], 'jpg');

        
end

