#include <zephyr/kernel.h>
#include <zephyr/bluetooth/bluetooth.h>
#include <zephyr/bluetooth/addr.h>
#include <zephyr/bluetooth/hci.h>
#include <zephyr/bluetooth/crypto.h>
#include <stdio.h>
#include <zephyr/logging/log.h>

uint8_t irk_key[16] = {0xDE, 0xAD, 0xBE, 0xEF, 0xDE, 0xAD, 0xBE, 0xEF, 0xDE, 0xAD, 0xBE, 0xEF, 0xDE, 0xAD, 0xBE, 0xEF};
uint8_t plaintext[16] __attribute__((section(".uninit")));
uint8_t ciphertext[16];

LOG_MODULE_REGISTER(rpa_attack_app);

int main(void){

	int id = bt_id_create(NULL, irk_key);
	//bt_id_reset(id, NULL, irk_key);

	struct bt_le_ext_adv *adv;
	const struct bt_le_adv_param adv_params = {
		.id = id,
		.options = BT_LE_ADV_OPT_CONN,
		.interval_min = BT_GAP_ADV_FAST_INT_MIN_2,
		.interval_max = BT_GAP_ADV_FAST_INT_MAX_2,
	};

	int err = bt_le_ext_adv_create(&adv_params, NULL, &adv);
	if(err){
		LOG_ERR("Falied to create advertising packet");
		return -1;
	}

	LOG_INF("Starting advertisement...");

	const struct bt_le_ext_adv_start_param ext_adv_params = {
		.timeout = 60,
		.num_events = 100,
	};

	err = bt_le_ext_adv_start(adv, &ext_adv_params);
	if(err){
		LOG_ERR("Failed to start advertising!");
		return -1;
	}

	k_sleep(K_SECONDS(60));

	bt_le_ext_adv_stop(adv);
	bt_le_ext_adv_delete(adv);

	return 0;	
}
