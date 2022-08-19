def get_default_config(data_name):
    if data_name in ['Caltech101-20']:
        return dict(
            seed=4,
            view=2,
            training=dict(
                start_dual_prediction=100,
                batch_size=256,
                epoch=500,
                alpha=10,
                lambda2=0.1,
                lambda1=0.1,
                lr=1.0e-4,
            ),
            Autoencoder=dict(
                arch1=[1984, 1024, 1024, 1024, 128],
                arch2=[512, 1024, 1024, 1024, 128],
                activations1='relu',
                activations2='relu',
                batchnorm=True,
            ),
            Prediction=dict(
                arch1=[128, 256, 128],
                arch2=[128, 256, 128],
            ),
        )

    elif data_name in ['Scene_15']:
        """The default configs."""
        return dict(
            seed=8,
            view=2,
            training=dict(
                lr=1.0e-4,
                start_dual_prediction=100,
                batch_size=256,
                epoch=500,
                alpha=10,
                lambda2=0.1,
                lambda1=0.1
            ),
            Autoencoder=dict(
                arch1=[20, 1024, 1024, 1024, 128],
                arch2=[59, 1024, 1024, 1024, 128],
                activations1='relu',
                activations2='relu',
                batchnorm=True,
            ),
            Prediction=dict(
                arch1=[128, 256, 128],
                arch2=[128, 256, 128],
            ),
        )

    elif data_name in ['NoisyMNIST']:
        """The default configs."""
        return dict(
            seed=1,
            view=2,
            Autoencoder=dict(
                arch1=[784, 1024, 1024, 1024, 40],
                arch2=[784, 1024, 1024, 1024, 40],
                activations1='relu',
                activations2='relu',
                batchnorm=True,
            ),
            Prediction=dict(
                arch1=[128, 256, 128],
                arch2=[128, 256, 128],
            ),
            training=dict(
                lr=1.0e-4,
                start_dual_prediction=100,
                batch_size=256,
                epoch=500,
                alpha=10,
                lambda1=0.1,
                lambda2=0.1,
            ),
        )

    elif data_name in ['LandUse_21']:
        """The default configs."""
        return dict(
            seed=4,
            view=2,
            Autoencoder=dict(
                arch1=[59, 1024, 1024, 1024, 40],
                arch2=[40, 1024, 1024, 1024, 40],
                activations1='relu',
                activations2='relu',
                batchnorm=True,
            ),
            Prediction=dict(
                arch1=[128, 256, 128],
                arch2=[128, 256, 128],
            ),
            training=dict(
                lr=1.0e-4,
                start_dual_prediction=100,
                batch_size=256,
                epoch=500,
                alpha=10,
                lambda2=0.1,
                lambda1=0.1,
            ),
        )

    else:
        raise Exception('Undefined data_name')
