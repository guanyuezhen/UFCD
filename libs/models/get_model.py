
def get_model_by_name(model_name, num_classes, in_width):
    if model_name == 'A2Net':
        from libs.models.A2Net.A2Net import get_model
        return get_model(bc_token_length=1, sc_token_length=num_classes)
    elif model_name == 'A2Net18':
        from libs.models.A2Net_18.A2Net import get_model
        return get_model(bc_token_length=1, sc_token_length=num_classes)
    elif model_name == 'BiSRNet':
        from libs.models.BiSRNet.BiSRNet import get_model
        return get_model(bc_token_length=1, sc_token_length=num_classes)
    elif model_name == 'SCanNet':
        from libs.models.SCanNet.SCanNet import get_model
        return get_model(bc_token_length=1, sc_token_length=num_classes, input_size=in_width)
    elif model_name == 'SSCDL':
        from libs.models.SSCDL.SSCDL import get_model
        return get_model(bc_token_length=1, sc_token_length=num_classes)
    elif model_name == 'TED':
        from libs.models.TED.TED import get_model
        return get_model(bc_token_length=1, sc_token_length=num_classes)
    else:
        from libs.models.A2Net.A2Net import get_model
        return get_model(bc_token_length=1, sc_token_length=num_classes)
