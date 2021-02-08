
function output = display_ranking (ranking, feat_names)

    n_feat = length (ranking);
%     array = cell (1,n_feat);
    for i =1:n_feat
        position = ranking (i);
        array (position) =  string(feat_names (i));
    end
    output = array;
%     disp (array);

end