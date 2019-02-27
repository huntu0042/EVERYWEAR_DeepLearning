
data_root = 'data/segment/';
% segment1 array, viton_segment_pair.txt/ you will make some txt
[segmentList] = textread('data/viton_segment_pairs.txt', '%s\n');

for i = 1:100
    segment_name = segmentList{i};
    % first segment size = 641 * 641
    
    segmentFile = load([data_root, segment_name]);
    segment = segmentFile.segment;
    
    % resize 641 * 644
    resize644 = zeros(641, 3);
    segment = [segment resize644];
    
    % manipulate model image position and segment mask
    addMatrix = zeros(20, 644);
    segment = [addMatrix;segment]; %661 * 644
    segment(484:661,:) = [];
    size(segment);
    
    % transpose
    segment = transpose(segment);
    save([data_root, segment_name],'segment');
end 
