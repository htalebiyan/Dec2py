% function [x,fval]=main(var_index, constr_rhs, constr_sense, obj_coeffs, x0)
rng default % For reproducibility
load('data.mat');
Asparse = load('arrdata.mat');
lambda = 1e5;
Afull = full(Asparse.A);
nvars = size(obj_coeffs,2);    % Number of variables
[ub,lb,IntCon] = bounds(var_index);
[Ai, bi, Ae, be] = seperate_constr(constr_rhs, constr_sense, Afull);
ObjectiveFunction = @(x)obj_function(x, obj_coeffs, Ai, bi, Ae, be, lambda);
initial_pop = cell2mat(struct2cell(x0))';
% initial_pop = initial_pop + randn(1,length(initial_pop)) .* 0001;

%% run the GA solver.
options = optimoptions('ga','PlotFcn', @gaplotbestf, ...
'UseParallel', true, 'EliteCount', 1000, 'PopulationSize', 5000, 'MaxGenerations', 2000);
% ,'InitialPopulationMatrix',initial_pop);
[x,fval,exitflag,output] = ga(ObjectiveFunction,nvars,Ai,bi,[],[],lb,ub,[],IntCon,options);

% [x,fval,exitflag,output] = intlinprog(cell2mat(obj_coeffs),IntCon,Ai,bi,Ae,be,lb,ub);

% [x,fval,exitflag,output] = particleswarm(ObjectiveFunction,nvars,lb,ub);

% options = optimoptions('simulannealbnd','PlotFcns',...
%           {@saplotbestx,@saplotbestf,@saplotx,@saplotf});
% [x,fval,exitflag,output] = simulannealbnd(ObjectiveFunction,initial_pop,lb,ub,options);


% disp(x);
disp(fval);
disp(exitflag);
disp(output);
disp(fval-sum(abs(Ae*x'-be))*lambda-sum(Ai*x'-bi));
disp(Ae*x'-be);
% disp(Ai*x'-bi);