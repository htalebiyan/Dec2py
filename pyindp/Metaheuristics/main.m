function [x,fval]=main(var_index, constr_rhs, constr_sense, obj_coeffs, x0)
rng default % For reproducibility
% load('data.mat');
Asparse = load('arrdata.mat');
lambda = 1e14;
Afull = full(Asparse.A);
nvars = size(obj_coeffs,2);    % Number of variables
[ub,lb,IntCon] = bounds(var_index);
[Ai, bi, Ae, be] = seperate_constr(constr_rhs, constr_sense, Afull);
ObjectiveFunction = @(x)obj_function(x, obj_coeffs, Ai, bi, Ae, be, lambda);
initial_pop = cell2mat(struct2cell(x0))';
% initial_pop = initial_pop + randn(1,length(initial_pop)) .* 001;

%% run the GA solver.
options = optimoptions('ga','PlotFcn', @gaplotbestf,'UseParallel', false,...
'EliteCount', 2, 'PopulationSize', 100, 'MaxGenerations', 200);
%     'InitialPopulationMatrix',initial_pop);
[x,fval,exitflag,output] = ga(ObjectiveFunction,nvars,Ai,bi,[],[],lb,ub,[],IntCon,options);

% [x,fval,exitflag,output] = intlinprog(cell2mat(obj_coeffs),IntCon,Ai,bi,Ae,be,lb,ub);

% [x,fval,exitflag,output] = particleswarm(ObjectiveFunction,nvars,lb,ub);

disp(x);
disp(fval);
disp(exitflag);
disp(output);
disp(fval-sum(abs(Ae*x'-be))*lambda);
disp(Ae*x'-be);