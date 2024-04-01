function [y, lambda, mu, sigma] = yeojohnson_transformation(x, eps, standardize)
% Yeo-Johnson Normalization
%  
% Yeo-Johnson transformation is a statistical method for normalizing data 
% and stabilizing variance. Unlike the Box-Cox transformation, it can handle 
% both positive and negative input values, offering greater flexibility. 
% This feature is especially beneficial for datasets with diverse value ranges, 
% aiding in achieving normality and variance stabilization essential for 
% statistical analysis.
%
% This function replicates yeojohnson R-function: it calculates the same lambda
% value, it applies the same transformation and it gives the same normalized values.
% https://github.com/petersonR/bestNormalize/blob/master/R/yeojohnson.R
%
% REQUIRED INPUTS:
%   x > matrix (n,1) with raw values
%
% OPTIONAL INPUTS:    
%   eps > threshold to compare lambda against (default: eps=0.001)
%   standardize > logical (true/false); if true (default),  the transformed values 
%                 are also centered and scaled, such that the transformation 
%                 attempts a standard normal
% OUTPUS:
%   y > transformed data with Yeo-Johnson Normalization
%   lambda > lambda value used during normalization
%   mu > mean of transformed data
%   sigma > standard deviation of transformed data
%
%
% Authors: Karel Mauricio Lopez-Vilaret, Marina Fernandez-Alvarez
% marina.fdez.alvarez@gmail.com


% Yeo-Johnson Normalization:
    if nargin < 2 || isempty(eps)
        eps = 0.001; % Threshold to treat lambda values close to zero or two
    end
    if nargin < 3 || isempty(standardize)
        standardize = true; % By default, it standardizes the output
    end

    % Estimation of the optimal lambda parameter
    lambda = estimateYeoJohnsonLambda(x, eps);

    % Application of the Yeo-Johnson transformation
    y = yeojohnsonTrans(x, lambda, eps);

    % Option to standardize transformed data
    mu = mean(y, 'omitnan');
    sigma = std(y, 'omitnan');
    if standardize
        y = (y - mu) / sigma;
    end

end


% MISC 
% Lambda estimation
function lambda = estimateYeoJohnsonLambda(x, eps, lower, upper)
    if nargin==1
            eps = 0.001;
            lower = -5;
            upper = 5;
    end
    if nargin==2
            lower = -5;
            upper = 5;
    end
    if nargin==1
            eps = 0.001;
            lower = -5;
            upper = 5;
    end

    % Data preparation
    x = x(~isnan(x)); % Excluyendo valores NaN como en R con !is.na(x)
    n = numel(x);
    constant = sum(sign(x) .* log(abs(x) + 1));
    
    % Función de log-verosimilitud
    function [loglik] = yjLogLik(lambda,x,n,constant, eps)
        x_t = yeojohnsonTrans(x, lambda, eps);
        x_t_var = var(x_t) * (n - 1) / n; % Ajuste para el cálculo de la varianza
        loglik = -0.5 * n * log(x_t_var) + (lambda - 1) * constant;
    end
    
    % Optimization to find the lambda that maximizes the log-likelihood
    options = optimset('TolX', 0.0001);
    [lambda, ~] = fminbnd(@(lambda) -yjLogLik(lambda,x,n,constant, eps), lower, upper, options);
end

% Yeojohnson Transformation
function x_t = yeojohnsonTrans(x, lambda, eps)

    % Data preparation
    x_t = zeros(size(x));
    x = x(~isnan(x)); % Excluyendo valores NaN como en R con !is.na(x)
    pos_idx = find(x >= 0);
    neg_idx = find(x < 0);
    
    % Transformation for positive values
    if ~isempty(pos_idx)
        x_pos = x(pos_idx);
        if abs(lambda) < eps
            x_t(pos_idx) = log(x_pos + 1);
        else
            x_t(pos_idx) = ((x_pos + 1).^lambda - 1) / lambda;
        end
    end
    % Transformation for negative values
    if ~isempty(neg_idx)
        x_neg = x(neg_idx);
        if abs(2 - lambda) < eps
            x_t(neg_idx) = -log(-x_neg + 1);
        else
            x_t(neg_idx) = -((-x_neg + 1).^(2 - lambda) - 1) / (2 - lambda);
        end
    end
end


