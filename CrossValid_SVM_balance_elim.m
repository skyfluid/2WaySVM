clc
clear
close all

def_bin_size = 50;
def_balance_coef = 1;

load(['../bindatagen/binMatWord_' int2str(def_bin_size) '.mat'])

wordtoVerify = {'CLAW' 'CRICKET' 'FLAG' 'FORK' 'LION' 'MEDAL' 'OYSTER' 'SERPENT' 'SHELF' 'SHIRT'};
mkdir('PCA_SVM_balance/')

trialSet = [];
wordSet = [];

for iTrial=1:length(binMat)
    A = binMat{iTrial}.mat';
    B = binMat{iTrial}.type;
    trialSet = [trialSet; A(:)'];
    wordSet = [wordSet; B];
end

outfile = fopen('datalog.txt','w');

stat_total = {};
for nPC=1:400
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
            svmStruct = svmtrain(scoreSel, distingSet, 'options', options); %, 'showplot',true);
            resultSet = svmclassify(svmStruct, verifySet*coeff(:, 1:nPC)); %, 'showplot',true);
            ansSet = strcmp(wordtoVerify{iWord}, verifySetw);
            hitRate = sum(eq(resultSet, ansSet))/length(ansSet);
            posHitRate = sum(resultSet(find(ansSet)))/length(find(ansSet));

            if (hitRate > local_best_score | ((hitRate == local_best_score) & (posHitRate > local_best_scorep)))
                local_best_score = hitRate;
                local_best_scorep = posHitRate;
                SVM_candidate{iTestSet}.svm = svmStruct;
                SVM_candidate{iTestSet}.pc = coeff(:, 1:nPC);
                SVM_candidate{iTestSet}.varcov = varCov;
            end
        end
        resultSet = svmclassify(SVM_candidate{iTestSet}.svm, testSet*SVM_candidate{iTestSet}.pc);
        ansSet = strcmp(wordtoVerify{iWord}, testSetw);
        hitRate = sum(eq(resultSet, ansSet))/length(ansSet);
        posHitRate = sum(resultSet(find(ansSet)))/length(find(ansSet));
        negHitRate = (length(find(ansSet==0)) - sum(resultSet(find(ansSet==0))))/length(find(ansSet==0));
        
        fprintf(outfile, 'Word: %s, SVM-%2d, #TestVec: %d, #HR: %f, #pHR: %f, #nHR: %f, #PC:%d\n', wordtoVerify{iWord}, iTestSet, length(ansSet), hitRate, posHitRate, negHitRate, nPC);
        stat_total{iWord}.data(nPC).HR(iTestSet) = hitRate;
        stat_total{iWord}.data(nPC).pHR(iTestSet) = posHitRate;
        stat_total{iWord}.data(nPC).nHR(iTestSet) = negHitRate;
        stat_total{iWord}.data(nPC).varcov(iTestSet) = SVM_candidate{iTestSet}.varcov;
        stat_total{iWord}.data(nPC).pc{iTestSet} = SVM_candidate{iTestSet}.pc;
        stat_total{iWord}.data(nPC).svm{iTestSet} = SVM_candidate{iTestSet}.svm;
    end
    stat_total{iWord}.avgHR(nPC) = mean(stat_total{iWord}.data(nPC).HR, 2);
    stat_total{iWord}.avgpHR(nPC) = mean(stat_total{iWord}.data(nPC).pHR, 2);
    stat_total{iWord}.avgnHR(nPC) = mean(stat_total{iWord}.data(nPC).nHR, 2);
    stat_total{iWord}.avgvarcov(nPC) = mean(stat_total{iWord}.data(nPC).varcov, 2);
end

save(['stat_total_balance_'  int2str(def_bin_size)  'ms.mat'], 'stat_total', '-v7.3')

end

fclose(outfile);

set(0, 'DefaultFigureVisible', 'on')

load(['stat_total_balance_'  int2str(def_bin_size)  'ms.mat']);

for iWord=1:length(wordtoVerify)
    avgHR = [];
    avgpHR = [];
    avgnHR = [];
    avgvarcov = [];
    dp = [];
    
    for nPC=1:28

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
    title(['Accuracy of Two-Way SVM (balance) for "' wordtoVerify{iWord} '" (199u 28pc)']);
    xlabel('Number of PCs'); ylabel('Accuracy');
    saveas(gcf, ['./PCA_SVM_balance/Accu_balance_' wordtoVerify{iWord} 'box.jpg'], 'jpg');

    figure;
    boxplot(avgpHR);
    title(['Accuracy (true) of Two-Way SVM (balance) for "' wordtoVerify{iWord} '" (199u 28pc)']);
    xlabel('Number of PCs'); ylabel('Accuracy: hit/ (hit+miss)');
    saveas(gcf, ['./PCA_SVM_balance/AccuP_balance_' wordtoVerify{iWord} '_box.jpg'], 'jpg');

    figure;
    boxplot(avgnHR);
    title(['Accuracy (false) of Two-Way SVM (balance) for "' wordtoVerify{iWord} '" (199u 28pc)']);
    xlabel('Number of PCs'); ylabel('Accuracy: CR/(CR+FA)');
    saveas(gcf, ['./PCA_SVM_balance/AccuN_balance_' wordtoVerify{iWord} '_box.jpg'], 'jpg');

    figure;
    boxplot(avgvarcov);
    title(['Variance Coverage of Two-Way SVM (balance) for "' wordtoVerify{iWord} '" (199u 28pc)']);
    xlabel('Number of PCs'); ylabel('Variance Coverage');
    saveas(gcf, ['./PCA_SVM_balance/Varcov_balance_' wordtoVerify{iWord} '_box.jpg'], 'jpg');    

    figure;
    boxplot(dp);
    title(['d-prime for word ' wordtoVerify{iWord} ' (199u 28pc) (balance)']);
    xlabel('Number of PCs'); ylabel('Variance Coverage');
    saveas(gcf, ['./PCA_SVM_balance/DP_balance_' wordtoVerify{iWord} '_box.jpg'], 'jpg');

        
end

