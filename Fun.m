%% No4_1，只聚类两次
%% DE/current-to-Cpbest/1, 其中(X_r1)、(X_r2)来自pos
function [gbest,gbestval, fitcount,RecordT,cSTART] = Fun(fhd,D, Max_nfe, XRmin, XRmax, subfolderPath, subfolder, targetbest,varargin)
stm = RandStream('swb2712','Seed',sum(100*clock));
RandStream.setGlobalStream(stm);
cSTART = 0;
% ps_ini = 18*D;
ps_ini = round(25*log(D)*sqrt(D));
ps_min = 5;
ps = ps_ini;

%% 详细执行时间报告
%  profile on
if length(XRmin) == 1
    Rmin = repmat(XRmin,1,D);
    Rmax = repmat(XRmax,1,D);
end
VRmin = repmat(Rmin,ps,1);
VRmax = repmat(Rmax,ps,1);

fidvec = cell2mat(varargin);
fid = fidvec(1);
runid = fidvec(2);
already_clustered = false;% 定义一个变量，用于跟踪是否已经执行过fitcount > 2/3 * Max_nfe的聚类
WriteFlag = true;
if WriteFlag
    name = fullfile(subfolderPath,[subfolder,'_fid_',num2str(fid),'_',num2str(D),'D_',num2str(runid),'.dat']);
    fout = fopen(name,'wt');
end
tic;

% 使用lhsdesign生成[0,1]范围内的拉丁超立方样本
lhs_samples = lhsdesign(ps, D);

% 将[0,1]范围的样本映射到[VRmin, VRmax]范围
pos = VRmin + (VRmax - VRmin) .* lhs_samples;

% 初始化时，在pos矩阵中为每个个体添加一个额外的维度作为簇标签，并将其设置为-1];
pos = [pos, -1 * ones(ps, 1)];
%给种群设置counter标签位
counter = zeros(ps,1);
pivot = [2/3*Max_nfe,1/3*ps_ini];
pastval = feval(fhd, pos(:, 1:D)', varargin{:});
[pastval , indexSel] = sort(pastval);
gbestval = pastval(1);
gbest = pos(indexSel(1),1:D);
fitcount = ps;

if WriteFlag
    fprintf(fout,'%d\t%.15f\n',1,gbestval-targetbest(fid));
end

% Cr = zeros(1,ps) + 0.9;
memory_size = 6; memory_order = 1;
memory_MUF = 0.5.*ones(memory_size,1); memory_MUCr = 0.9.*ones(memory_size,1);
memory_MUF_elite = 0.5.*ones(memory_size,1); memory_MUCr_elite = 0.9.*ones(memory_size,1);
Archfactor = 1.05 + abs(D - 30) * 11/400;

% pbest_rate_max = 0.2; pbest_rate_min = 0.05;
pbest_rate = 0.15;
decayA = [];
T0 = 1.0;
% decayRate = T0/56;
A = [];
A_eval = [];
gen = 1;


Xmax = 100;Xmin = -100;
n = D;
Mean1 = mean(pos);
DI_ini = sqrt(sum(pdist2(Mean1,pos)));
RDI = DI_ini;
t1 = repmat(0.9,ps,1);

%% 聚类设置
ParameterCase;% 调用 ParameterCase 脚本来获取参数
EliteLayering = true;%是否执行聚类
case_idx = 0;
currentParameterCase = eval(['ParameterCase', num2str(case_idx)]);

if fitcount < ceil(pivot(1))
    num_clusters = currentParameterCase(1, 1);
    Kmeans = currentParameterCase(1, 2);
else
    num_clusters = currentParameterCase(2, 1);
    Kmeans = currentParameterCase(2, 2);
end

%% 开始进化
while fitcount < Max_nfe

    %% 聚类
    % 当进入新的迭代或检查fitcount时
    if gen == 1 || (fitcount > 2/3 * Max_nfe && ~already_clustered)
        [cluster_idx, cluster_centers] = kmeans(pos(:, 1:end-1), num_clusters);

        % 更新簇标签
        pos(:, end) = cluster_idx;
        if gen == 1
            gbest_cluster = pos(indexSel(1),end);
        end
        % 如果基于fitcount的条件被满足，更新already_clustered变量以防止再次执行聚类
        if fitcount > 2/3 * Max_nfe
            already_clustered = true;
        end
    end

    % 如果超过指定的个体数量，删除多余个体
    if gen>1
        % 适应度排序，并获取排序索引
        [~, indexSel] = sort(pastval);
        gbest = pos(indexSel(1),1:D);
        gbestval = pastval(indexSel(1));
        gbest_cluster = pos(indexSel(1),end);

        selected_indices = indexSel(1:ps);
        pos = pos(selected_indices, :);
        pastval = pastval(selected_indices);
        counter = counter(selected_indices);
        Cr = [];F = [];  Cr_elite = [];  F_elite= [];
        VRmin = repmat(Rmin,ps,1);VRmax = repmat(Rmax,ps,1);
    end




    %% pbest集合:pbest集改为当前最优个体所在簇中适应度较高的邻居
    %     [~, indexSel] = sort(pastval);

    pNP = max(round(pbest_rate*ps),2);

    % 在gbest所在的簇中选择适应度最小的邻居
    indices_in_gbest_cluster = find(pos(:, end) == gbest_cluster);

    % 获取这些邻居的适应度值
    fitness_values_of_gbest_cluster = pastval(indices_in_gbest_cluster);

    % 对这些适应度值进行排序并获取排序索引
    [~, sorted_indices] = sort(fitness_values_of_gbest_cluster);

    % 选择适应度前10%的个体索引
    half_length = round(length(sorted_indices) * 0.10);

    selected_indices = indices_in_gbest_cluster(sorted_indices(1:half_length));

    next_index = 2;
    % 如果half_length个数小于所需要的pNP数值，则按照适应度由低到高排序添加适应度较小的个体，主要不要添加重复个体
    while half_length < pNP
        if ~ismember(next_index, selected_indices)
            selected_indices = [selected_indices; next_index];
            next_index = next_index + 1;
            half_length = half_length + 1;
        else
            next_index = next_index + 1;
        end
    end
    Restart_val = pastval(pNP);

    % 将gbest和其领居组合成一个优质种群
    Cpnei0 = [pos(selected_indices, :)];
    lenSel = max(round(pNP*rand(1,ps)),1);
    Cpnei = Cpnei0(lenSel,:);



    %% A集合
    unique_clusters = unique(pos(:, end)); % 获取所有独特的簇标签

    % 遍历每个簇标签
    for cluster_label = unique_clusters'
        % 找到所有属于当前簇的个体的索引
        indices_in_cluster = find(pos(:, end) == cluster_label);

        % 计算要选择的个体数量
        num_to_select = ceil(Kmeans * length(indices_in_cluster));

        % 随机选择个体
        random_selection = indices_in_cluster(randperm(length(indices_in_cluster), num_to_select));
        % 将随机选择的个体添加到A中

        ADD_A = pos(random_selection, :);
        A = [A ; ADD_A];

    end
    [~, index] = unique(A, 'rows', 'stable'); % 找到唯一行的索引
    A = A(index, :); % 仅保留唯一的行
    psExt = size(A,1);


    %% posr、posxr集合
    rndBase = randperm(ps)';
    %     rndSeq1 = ceil(rand(ps,1)*ps);
    rndSeq1 = ceil(rand(ps,1)*psExt);
    for ii = 1:ps
        while rndBase(ii)==ii
            rndBase(ii)=ceil(rand()*ps);
        end
        while rndSeq1(ii) == ii %|| rndSeq1(ii) == rndBase(ii)
            rndSeq1(ii) = ceil(rand()*psExt);
        end
    end
    posr = pos(rndBase,:);
    %     posxr = pos(rndSeq1,:);
    posxr = A(rndSeq1,:);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%       F/Cr生成公式       %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    memory_rand_index1 = ceil(memory_size*rand(ps,1));
    MUF = memory_MUF(memory_rand_index1);
    MUCr = memory_MUCr(memory_rand_index1);

    %% for generating crossover rate Cr
    rand1 = rand(ps,1);
    rand2 = rand(ps,1);
    for ii = 1:ps
        if rand1(ii) < t1(ii)
            Cr(ii) = normrnd(MUCr(ii),0.1);
        else
            Cr(ii) = 0.9;
        end
        if find(MUCr(ii) == -1)
            Cr(ii) = 0;
        end
    end
    Cr = min(Cr, 1);Cr = max(Cr, 0);

    %% for generating scal factor F
    for ii = 1:ps
        F(ii,1) = randCauchy(MUF(ii),0.1);
    end
    label_sf = find(F <= 0);
    while ~ isempty(label_sf)
        F(label_sf) = MUF(label_sf) + 0.1 * tan(pi * (rand(length(label_sf), 1) - 0.5));
        label_sf = find(F <= 0);
    end

    F = min(F,1);

    %% for generating label
    label=zeros(ps,D);
    rndVal = rand(ps,D);
    onemat = zeros(ps,D);
    for ii = 1:ps
        label(ii,:) = rndVal(ii,:)<=Cr(ii);
        indexJ = ceil(rand()*D);
        onemat(ii,indexJ) = 1;
    end
    label = label|onemat;

    %% 变异策略（第二）
    %% offset = F.*(pbestB-pos)+F.*(posr-posxr);
    offset = F .* (Cpnei(:, 1:D) - pos(:, 1:D)) + F .* (posr(:, 1:D) - posxr(:, 1:D));
    copy_pos = pos;

    % 变异
    pos(:, 1:D) = copy_pos(:, 1:D) + offset;

    % 交叉
    label = [label,zeros(size(label,1),1)];
    pos(~label) = copy_pos(~label); %交叉

    % 边界处理
    pos(:, 1:D) = ((pos(:, 1:D) >= VRmin) & (pos(:, 1:D) <= VRmax)) .* pos(:, 1:D) ...
        + (pos(:, 1:D) < VRmin) .* ((VRmin + copy_pos(:, 1:D)) .* rand(ps, D) / 2) ...
        + (pos(:, 1:D) > VRmax) .* ((VRmax + copy_pos(:, 1:D)) .* rand(ps, D) / 2);

    % dis
    dis = offset .* label(:, 1:D);

    posval = feval(fhd, pos(:, 1:D)', varargin{:});
    [posval, I] = min([posval ; pastval], [], 1);% I=1表示更新之后更好 ； I=2表示更新之后变差
    pos(I == 2,:) = copy_pos(I == 2,:); %选择
    copy_pos = [];
    counter(I == 2) = counter(I == 2) + 1;
    counter(I == 1) = 0;

    %计算主种群RDI
    Meanpos = mean(pos);
    DI = sqrt(sum(pdist2(Meanpos,pos)));
    RDI1 = DI/DI_ini;
    RDI = [RDI;RDI1];
    fitcount = fitcount + ps;

    len = length(A(:,1));
    if len>round(Archfactor*ps)
        rndSel = randperm(len)';
        rndSel = rndSel(round(Archfactor*ps)+1:len);
        A(rndSel,:) = [];
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%       F\Cr参数自适应更新      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    SuccF = F(I==1);
    SuccCr = Cr(I==1);
    dis = dis(I==1,:);
    FCR = std(dis,0,2);
    FCR = FCR/sum(FCR);
    %% 自适应更新memory_MUF、memory_MUCr、memory_MUF_elite
    num_Succ = numel(SuccCr);
    if num_Succ > 0
        memory_MUF(memory_order) = (FCR'*(SuccF.^2))/(FCR'*SuccF);
        if max(SuccCr) == 0 || memory_MUCr(memory_order) == -1
            memory_MUCr(memory_order) = -1;
        else
            memory_MUCr(memory_order) = ((SuccCr.^2)*FCR)/(SuccCr*FCR);
        end
    end

    memory_order = memory_order + 1;
    if memory_order > memory_size
        memory_order = 1;
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%      找gbest、gbestval      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    pastval = posval;
    posval = [];
    [gbestval,gbestid] = min(pastval);
    gbest = pos(gbestid,1:D);

    if((mod(fitcount-ps,100*5*D)>=100*5*D-ps)&& mod(fitcount,100*5*D)<ps)  &&  WriteFlag
        fprintf(fout,'%d\t%.15f\n',fitcount,gbestval-targetbest(fid));
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%      重启机制      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% restart mechanism
    V_STD = std(pos(:,1:D));
    [~,v_label] = sort(V_STD,'descend');
    v_label = v_label(1:ceil(0.3*D));

    if RDI1 < 0.01
        for i = 1:length(counter)
            if counter(i)>=n && i~=gbestid && pastval(i)>Restart_val
%                 if cSTART <= 3000
%                                     pos(i,v_label) = pos(i,v_label) + 0.5 *rand(1,length(v_label)).* ( gbest(v_label) - Meanpos(v_label) );
                    pos(i,v_label) = pos(i,v_label) + F(i) .* (Cpnei(i,v_label) - pos(i,v_label)) + F(i) .* (posr(i,v_label) - posxr(i,v_label));
                    %                 pastval(i) = feval(fhd,pos(i,1:D)',fid);
                    %                 fitcount=fitcount+1;
%                     START = START+1;
%                 end
            elseif counter(i)>=n+15 && i~=gbestid
                %                 pos(i,v_label) = gbest(v_label) + 0.1 *rand(1,length(v_label)).* ( gbest(v_label) - Meanpos(v_label) );
                %                pos(i,v_label) = pos(i,v_label) + F(i) .* (Cpnei(i,v_label) - pos(i,v_label)) + F(i) .* (posr(i,v_label) - posxr(i,v_label));
%                 pos(i,1:D) = pos(i,1:D) + F(i) .* (gbest - pos(i,1:D)) + F(i) .* (posr(i,1:D) - posxr(i,1:D));
                pos(i,1:D) = gbest + F(i) .* (posr(i,1:D) - posxr(i,1:D));
                pastval(i) = feval(fhd,pos(i,1:D)',fid);
                counter(i)=0;
                fitcount=fitcount+1;
                cSTART = cSTART+1;
            end
        end
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%      种群递减策略      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if fitcount<1/3*Max_nfe
        plan_ps =  ceil((pivot(2)-ps_ini)/(pivot(1)-ps_ini)^2*(fitcount-ps_ini)^2 + ps_ini);
    elseif fitcount<ceil(pivot(1))
        plan_ps = ceil((pivot(2)-ps_ini)/(pivot(1)-ps_ini)^2*(fitcount-ps_ini)^2 + ps_ini);
    else
        plan_ps =  floor((pivot(2)-ps_ini)/(pivot(1)-ps_ini)*(fitcount-Max_nfe)+ps_min);
    end
    % plan_ps = round((((ps_min - ps_ini) / Max_nfe) * fitcount) + ps_ini);

    %% 记录ps变化情况
    %         reduce_ps = fullfile(subfolderPath,['TEST03_', num2str(fff), '_reduce-ps.txt']);
    %         f_reduce = fopen(reduce_ps,'a');
    %        fprintf(f_reduce,'%d\n',reduc_nums);
    if ps > plan_ps
        if plan_ps < ps_min
            ps = ps_min;
        else
            ps = plan_ps;
        end
    end
    %     pbest_rate = pbest_rate_max+(pbest_rate_min-pbest_rate_max)*fitcount/Max_nfe;
    pbest_rate =  0.15 + 0.3*fitcount/Max_nfe;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%      种群递减策略结束      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    gen = gen +1;
end
RecordT = toc;

%% 详细执行时间报告
%  profile off
%profile viewer

if WriteFlag
    fclose(fout);
end

end

function result = randCauchy(mu, sigma)
[m,n] = size(mu);
result = mu + sigma*tan(pi*(rand(m,n)-0.5));
end