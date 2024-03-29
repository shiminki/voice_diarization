import torch
from nemo.collections.asr.parts.utils.longform_clustering import LongFormSpeakerClustering
from nemo.collections.asr.parts.utils.offline_clustering import (
    SpeakerClustering,
    NMESC,
    SpectralClustering,
    getAffinityGraphMat,
    estimateNumofSpeakers
)

class LongFormSpeakerClusteringCuda(LongFormSpeakerClustering):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.cuda = device.type == 'cuda'
        self.speaker_clustering = SpeakerClusteringCuda(device=device)

    def forward_unit_infer(
        self,
        mat: torch.Tensor,
        oracle_num_speakers: int = -1,
        max_num_speakers: int = 8,
        max_rp_threshold: float = 0.15,
        sparse_search_volume: int = 30,
        est_num_of_spk_enhanced: torch.Tensor = torch.tensor(-1),
        fixed_thres: float = -1.0,
        kmeans_random_trials: int = 1,
    ) -> torch.LongTensor:
        """
        This function takes a cosine similarity matrix `mat` and returns the speaker labels for the segments 
        in the given input embeddings. 
       
        Args: 
            mat (Tensor):
                Cosine similarity matrix (affinity matrix) calculated from the provided speaker embeddings.
            oracle_num_speakers (int):
                The number of speakers in a session, as specified by the reference transcript.
                Can be used as `chunk_cluster_count` in long-form clustering mode.
            max_num_speakers (int):
                The upper bound for the number of speakers in each session.
            max_rp_threshold (float):
                Limits the range of parameter search.
                The clustering performance can vary based on this range.
                The default value is 0.15.
            sparse_search_volume (int):
                The number of p_values considered during NME analysis.
                The default is 30. Lower values speed up the NME-analysis but might lead to poorer parameter estimations. Values below 20 are not recommended.
            est_num_of_spk_enhanced (int):
                The number of speakers estimated from enhanced speaker counting.
                If the value is -1, the enhanced speaker counting is skipped.
            fixed_thres (float):
                If a `fixed_thres` value is provided, the NME-analysis process will be skipped.
                This value should be optimized on a development set for best results.
                By default, it is set to -1.0, and the function performs NME-analysis to estimate the threshold.
            kmeans_random_trials (int):
                The number of random trials for initializing k-means clustering. More trials can result in more stable clustering. The default is 1. 
                
        Returns:
            Y (LongTensor):
                Speaker labels (clustering output) in integer format for the segments in the given input embeddings.
        """
        nmesc = NMESCCuda(
            mat,
            max_num_speakers=max_num_speakers,
            max_rp_threshold=max_rp_threshold,
            sparse_search=self.sparse_search,
            sparse_search_volume=sparse_search_volume,
            fixed_thres=fixed_thres,
            nme_mat_size=self.nme_mat_size,
            maj_vote_spk_count=self.maj_vote_spk_count,
            parallelism=self.parallelism,
            cuda=self.cuda,
            device=self.device,
        )
        # If there are less than `min_samples_for_nmesc` segments, est_num_of_spk is 1.
        if mat.shape[0] > self.min_samples_for_nmesc:
            est_num_of_spk, p_hat_value = nmesc.forward()
            affinity_mat = getAffinityGraphMat(mat, p_hat_value)
        else:
            nmesc.fixed_thres = max_rp_threshold
            est_num_of_spk, p_hat_value = nmesc.forward()
            affinity_mat = mat

        # `n_clusters` is number of speakers estimated from spectral clustering.
        if oracle_num_speakers > 0:
            n_clusters = int(oracle_num_speakers)
        elif est_num_of_spk_enhanced > 0:
            n_clusters = int(est_num_of_spk_enhanced.item())
        else:
            n_clusters = int(est_num_of_spk.item())

        spectral_model = SpectralClustering(
            n_clusters=n_clusters, n_random_trials=kmeans_random_trials, cuda=self.cuda, device=self.device
        )
        Y = spectral_model.forward(affinity_mat)
        return Y


    def _forward_infer(
        self,
        embeddings_in_scales: torch.Tensor,
        timestamps_in_scales: torch.Tensor,
        multiscale_segment_counts: torch.LongTensor,
        multiscale_weights: torch.Tensor,
        oracle_num_speakers: int = -1,
        max_rp_threshold: float = 0.15,
        max_num_speakers: int = 8,
        enhanced_count_thres: int = 80,
        sparse_search_volume: int = 30,
        fixed_thres: float = -1.0,
        chunk_cluster_count: int = 50,
        embeddings_per_chunk: int = 10000,
    ) -> torch.LongTensor:
        """
        This function is a wrapper designed for toggling between long-form and short-form speaker clustering.
        The details of short-form clustering is in `SpeakerClustering` class.
        NOTE: `torch.jit.script` currently does not support `**kwargs` in the function signature therefore,
        we need to use a wrapper function to handle the arguments.
        """
        if embeddings_per_chunk is not None and torch.max(multiscale_segment_counts) > embeddings_per_chunk:
            return self.long_forward_infer(
                embeddings_in_scales=embeddings_in_scales,
                timestamps_in_scales=timestamps_in_scales,
                multiscale_segment_counts=multiscale_segment_counts,
                multiscale_weights=multiscale_weights,
                oracle_num_speakers=oracle_num_speakers,
                max_rp_threshold=max_rp_threshold,
                max_num_speakers=max_num_speakers,
                sparse_search_volume=sparse_search_volume,
                fixed_thres=fixed_thres,
                chunk_cluster_count=chunk_cluster_count,
                embeddings_per_chunk=embeddings_per_chunk,
            )
        else:
            cluster_labels = self.speaker_clustering.forward_infer(
                embeddings_in_scales=embeddings_in_scales,
                timestamps_in_scales=timestamps_in_scales,
                multiscale_segment_counts=multiscale_segment_counts,
                multiscale_weights=multiscale_weights,
                oracle_num_speakers=oracle_num_speakers,
                max_rp_threshold=max_rp_threshold,
                max_num_speakers=max_num_speakers,
                enhanced_count_thres=enhanced_count_thres,
                sparse_search_volume=sparse_search_volume,
                fixed_thres=fixed_thres,
            )
            self.timestamps_in_scales = self.speaker_clustering.timestamps_in_scales
            return cluster_labels


class SpeakerClusteringCuda(SpeakerClustering):
    def __init__(
        self,
        min_samples_for_nmesc: int = 6,
        nme_mat_size: int = 512,
        sparse_search: bool = True,
        maj_vote_spk_count: bool = False,
        parallelism: bool = False,
        device = torch.device('cpu')
    ):
        super().__init__(
            min_samples_for_nmesc, nme_mat_size, sparse_search, maj_vote_spk_count,
            parallelism, device.type == 'cuda'
        )
        self.device = device
    
    def forward_unit_infer(
        self,
        mat: torch.Tensor,
        oracle_num_speakers: int = -1,
        max_num_speakers: int = 8,
        max_rp_threshold: float = 0.15,
        sparse_search_volume: int = 30,
        est_num_of_spk_enhanced: torch.Tensor = torch.tensor(-1),
        fixed_thres: float = -1.0,
        kmeans_random_trials: int = 1,
    ) -> torch.LongTensor:
        """
        This function takes a cosine similarity matrix `mat` and returns the speaker labels for the segments 
        in the given input embeddings. 
       
        Args: 
            mat (Tensor):
                Cosine similarity matrix (affinity matrix) calculated from the provided speaker embeddings.
            oracle_num_speakers (int):
                The number of speakers in a session, as specified by the reference transcript.
                Can be used as `chunk_cluster_count` in long-form clustering mode.
            max_num_speakers (int):
                The upper bound for the number of speakers in each session.
            max_rp_threshold (float):
                Limits the range of parameter search.
                The clustering performance can vary based on this range.
                The default value is 0.15.
            sparse_search_volume (int):
                The number of p_values considered during NME analysis.
                The default is 30. Lower values speed up the NME-analysis but might lead to poorer parameter estimations. Values below 20 are not recommended.
            est_num_of_spk_enhanced (int):
                The number of speakers estimated from enhanced speaker counting.
                If the value is -1, the enhanced speaker counting is skipped.
            fixed_thres (float):
                If a `fixed_thres` value is provided, the NME-analysis process will be skipped.
                This value should be optimized on a development set for best results.
                By default, it is set to -1.0, and the function performs NME-analysis to estimate the threshold.
            kmeans_random_trials (int):
                The number of random trials for initializing k-means clustering. More trials can result in more stable clustering. The default is 1. 
                
        Returns:
            Y (LongTensor):
                Speaker labels (clustering output) in integer format for the segments in the given input embeddings.
        """
        nmesc = NMESCCuda(
            mat,
            max_num_speakers=max_num_speakers,
            max_rp_threshold=max_rp_threshold,
            sparse_search=self.sparse_search,
            sparse_search_volume=sparse_search_volume,
            fixed_thres=fixed_thres,
            nme_mat_size=self.nme_mat_size,
            maj_vote_spk_count=self.maj_vote_spk_count,
            parallelism=self.parallelism,
            cuda=self.cuda,
            device=self.device,
        )
        # If there are less than `min_samples_for_nmesc` segments, est_num_of_spk is 1.
        if mat.shape[0] > self.min_samples_for_nmesc:
            est_num_of_spk, p_hat_value = nmesc.forward()
            affinity_mat = getAffinityGraphMat(mat, p_hat_value)
        else:
            nmesc.fixed_thres = max_rp_threshold
            est_num_of_spk, p_hat_value = nmesc.forward()
            affinity_mat = mat

        # `n_clusters` is number of speakers estimated from spectral clustering.
        if oracle_num_speakers > 0:
            n_clusters = int(oracle_num_speakers)
        elif est_num_of_spk_enhanced > 0:
            n_clusters = int(est_num_of_spk_enhanced.item())
        else:
            n_clusters = int(est_num_of_spk.item())

        spectral_model = SpectralClustering(
            n_clusters=n_clusters, n_random_trials=kmeans_random_trials, cuda=self.cuda, device=self.device
        )
        Y = spectral_model.forward(affinity_mat)
        return Y


class NMESCCuda(NMESC):
    def getEigRatio(self, p_neighbors: int) -> torch.Tensor:
        """
        For a given p_neighbors value, calculate g_p, which is a ratio between p_neighbors and the
        maximum eigengap values.
        References:
            Tae Jin Park et al., Auto-Tuning Spectral Clustering for Speaker Diarization Using
            Normalized Maximum Eigengap, IEEE Signal Processing Letters 27 (2019),
            https://arxiv.org/abs/2003.02405

        Args:
            p_neighbors (int):
                Determines how many binary graph connections we want to keep for each row.

        Returns:
            est_num_of_spk (int):
                Estimated number of speakers
            g_p (float):
                The ratio between p_neighbors value and the maximum eigen gap value.
        """
        affinity_mat = getAffinityGraphMat(self.mat, p_neighbors)
        est_num_of_spk, lambdas, lambda_gap_list = estimateNumofSpeakers(
            affinity_mat, self.max_num_speakers, self.cuda
        )
        arg_sorted_idx = torch.argsort(lambda_gap_list[: self.max_num_speakers], descending=True)
        max_key = arg_sorted_idx[0]
        max_eig_gap = lambda_gap_list[max_key] / (torch.max(lambdas).item() + self.eps)
        g_p = (p_neighbors / self.mat.shape[0]) / (max_eig_gap + self.eps)
        return torch.stack([g_p, est_num_of_spk])